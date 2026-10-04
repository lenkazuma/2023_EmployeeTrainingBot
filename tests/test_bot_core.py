import sys
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot_core import (  # noqa: E402
    EmptyDocumentError,
    answer,
    extract_pages,
    fingerprint,
    load_or_build_index,
    split_pages,
    summarize_conversation,
    summarize_document,
)

CHUNKS = [
    Document(page_content="营业收入为 100 亿元。", metadata={"page": 3}),
    Document(page_content="新员工入职需完成安全培训。", metadata={"page": 8}),
]


class CountingEmbedding(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def test_blank_pdf_raises():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(EmptyDocumentError):
        extract_pages(buffer.getvalue())


def test_split_keeps_page_numbers():
    chunks = split_pages([Document(page_content="培训内容。" * 200, metadata={"page": 4})], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1 and all(c.metadata["page"] == 4 for c in chunks)


def test_index_is_cached_per_document(tmp_path):
    embeddings = CountingEmbedding(size=8)
    cache = tmp_path / f"{fingerprint(b'doc')}.json"
    load_or_build_index(CHUNKS, embeddings, cache)
    reloaded = load_or_build_index(CHUNKS, embeddings, cache)
    assert embeddings.calls == 1
    assert len(reloaded.similarity_search("培训", k=2)) == 2


def test_different_documents_get_different_cache_keys():
    assert fingerprint(b"doc-a") != fingerprint(b"doc-b")


def test_answer_returns_pages_and_sends_history():
    store = load_or_build_index(CHUNKS, DeterministicFakeEmbedding(size=8), None)
    llm = FakeListChatModel(responses=["需要完成安全培训（第 8 页）"])
    result = answer("入职要做什么？", store, llm, "员工手册", history=[("你好", "你好！")], k=2)
    assert result.text.startswith("需要完成安全培训")
    assert result.pages == [3, 8]


def test_summaries():
    store = load_or_build_index(CHUNKS, DeterministicFakeEmbedding(size=8), None)
    llm = FakeListChatModel(responses=["文档摘要", "对话总结"])
    assert summarize_document(store, llm, "员工手册") == "文档摘要"
    assert summarize_conversation([("问", "答")], llm, "员工手册") == "对话总结"
    assert "没有对话" in summarize_conversation([], llm, "员工手册")
