"""Loading, indexing and Q&A for the training-manual chatbot, independent of Streamlit."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SYSTEM_PROMPT = (
    "你是一名叫 Ernie 的员工培训助手，正在阅读文档《{title}》。"
    "只根据提供的文档片段回答问题，片段标注了页码，回答时用（第 N 页）注明出处。"
    "如果片段中没有答案，就直接说不知道，不要编造。请始终使用中文回答。"
)
DOC_SUMMARY_PROMPT = "请根据以下《{title}》的内容片段，用中文写一份简洁的文档摘要。\n\n{context}"
CHAT_SUMMARY_PROMPT = (
    "以下是员工围绕《{title}》的问答记录，请用中文总结这次对话涉及的要点和结论。\n\n{history}"
)


class EmptyDocumentError(ValueError):
    """Raised when no text could be extracted from the PDF."""


@dataclass
class Answer:
    text: str
    pages: list[int]


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def download_pdf(url: str, timeout: float = 60) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def extract_pages(pdf_bytes: bytes) -> list[Document]:
    reader = PdfReader(BytesIO(pdf_bytes))
    docs = [
        Document(page_content=text, metadata={"page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]
    if not docs:
        raise EmptyDocumentError("没有从 PDF 中提取到文字，可能是扫描件。")
    return docs


def split_pages(docs: list[Document], chunk_size: int = 384, chunk_overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def load_or_build_index(chunks: list[Document], embeddings: Embeddings, cache_path: Path | None) -> InMemoryVectorStore:
    """Reuse a saved index for the same document so embeddings are paid for once."""
    if cache_path is not None and cache_path.exists():
        return InMemoryVectorStore.load(str(cache_path), embeddings)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        store.dump(str(cache_path))
    return store


def _context(docs: list[Document]) -> str:
    return "\n\n".join(f"[第 {d.metadata.get('page', '?')} 页]\n{d.page_content}" for d in docs)


def answer(
    question: str,
    store: InMemoryVectorStore,
    llm: BaseChatModel,
    title: str,
    history: list[tuple[str, str]] | None = None,
    k: int = 4,
    max_history: int = 5,
) -> Answer:
    docs = store.similarity_search(question, k=k)
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT.format(title=title))]
    for past_q, past_a in (history or [])[-max_history:]:
        messages += [("human", past_q), ("ai", past_a)]
    messages.append(("human", f"文档片段：\n{_context(docs)}\n\n问题：{question}"))
    pages = sorted({int(d.metadata.get("page", 0)) for d in docs})
    return Answer(text=str(llm.invoke(messages).content), pages=pages)


def summarize_document(store: InMemoryVectorStore, llm: BaseChatModel, title: str, k: int = 6) -> str:
    docs = store.similarity_search(f"{title} 主要内容 核心要点 概述", k=k)
    return str(llm.invoke(DOC_SUMMARY_PROMPT.format(title=title, context=_context(docs))).content)


def summarize_conversation(history: list[tuple[str, str]], llm: BaseChatModel, title: str) -> str:
    if not history:
        return "还没有对话内容可以总结。"
    transcript = "\n\n".join(f"问题：{q}\n回答：{a}" for q, a in history)
    return str(llm.invoke(CHAT_SUMMARY_PROMPT.format(title=title, history=transcript)).content)
