import os
from pathlib import Path

import streamlit as st
from dotenv import find_dotenv, load_dotenv

from bot_core import (
    EmptyDocumentError,
    answer,
    download_pdf,
    extract_pages,
    fingerprint,
    load_or_build_index,
    split_pages,
    summarize_conversation,
    summarize_document,
)

DEFAULT_TITLE = "万科企业股份有限公司2023年第一季度报告"
DEFAULT_URL = "http://static.cninfo.com.cn/finalpage/2023-04-29/1216686497.PDF"
MODELS = ["ERNIE-3.5-8K", "ERNIE-4.0-8K", "ERNIE-Speed-8K", "ERNIE-Lite-8K"]
INDEX_DIR = Path(".index_cache")
CHAT_CONTEXT_LENGTH = 10

load_dotenv(find_dotenv(), override=True)
st.set_page_config(page_title="员工培训助手", page_icon="🏠")


def credentials() -> dict:
    ak = st.session_state.get("qianfan_ak") or os.getenv("QIANFAN_AK", "")
    sk = st.session_state.get("qianfan_sk") or os.getenv("QIANFAN_SK", "")
    return {"qianfan_ak": ak, "qianfan_sk": sk} if ak and sk else {}


def make_llm(model: str):
    from langchain_community.chat_models import QianfanChatEndpoint

    return QianfanChatEndpoint(model=model, temperature=0.1, **credentials())


def make_embeddings():
    from langchain_community.embeddings import QianfanEmbeddingsEndpoint

    return QianfanEmbeddingsEndpoint(**credentials())


with st.sidebar:
    st.header("设置")
    st.caption("留空时使用环境变量 QIANFAN_AK / QIANFAN_SK（或 QIANFAN_ACCESS_KEY / QIANFAN_SECRET_KEY）。")
    st.text_input("千帆 API Key (AK)", type="password", key="qianfan_ak")
    st.text_input("千帆 Secret Key (SK)", type="password", key="qianfan_sk")
    model = st.selectbox("模型", MODELS)

    st.subheader("培训文档")
    source = st.radio("文档来源", ["示例：万科 2023 一季报", "上传 PDF", "PDF 链接"])
    uploaded = st.file_uploader("上传 PDF", type=["pdf"]) if source == "上传 PDF" else None
    url = st.text_input("PDF 链接", value="") if source == "PDF 链接" else DEFAULT_URL
    title = st.text_input("文档标题", value=DEFAULT_TITLE if source.startswith("示例") else "")

if source == "上传 PDF" and uploaded is None:
    st.info("请在左侧上传一份 PDF 培训文档。")
    st.stop()
if source == "PDF 链接" and not url:
    st.info("请在左侧填写 PDF 链接。")
    st.stop()

title = title or (uploaded.name if uploaded else "培训文档")
st.subheader(title)


@st.cache_data(show_spinner=False)
def fetch_pdf(pdf_url: str) -> bytes:
    return download_pdf(pdf_url)


try:
    with st.spinner("正在加载文档..."):
        pdf_bytes = uploaded.getvalue() if uploaded else fetch_pdf(url)
except Exception as exc:
    st.error(f"下载文档失败：{exc}")
    st.stop()

doc_id = fingerprint(pdf_bytes)
if st.session_state.get("doc_id") != doc_id:
    try:
        with st.spinner("正在建立文档索引（同一文档只需一次）..."):
            chunks = split_pages(extract_pages(pdf_bytes))
            store = load_or_build_index(chunks, make_embeddings(), INDEX_DIR / f"{doc_id}.json")
    except EmptyDocumentError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"建立索引失败，请检查千帆凭证：{exc}")
        st.stop()
    st.session_state.update(doc_id=doc_id, store=store, summary=None, history=[])

llm = make_llm(model)

if st.session_state.summary is None:
    try:
        with st.spinner("正在生成文档摘要..."):
            st.session_state.summary = summarize_document(st.session_state.store, llm, title)
    except Exception as exc:
        st.warning(f"生成摘要失败：{exc}")
with st.expander("文档摘要", expanded=True):
    st.write(st.session_state.summary or "")

st.chat_message("assistant").write("你好，我是文心智能助理 Ernie。请问你有什么问题呢？")
for turn in st.session_state.history:
    st.chat_message("user").write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["pages"]:
            st.caption("参考页码：" + "、".join(str(p) for p in turn["pages"]))

question = st.chat_input("请输入你的问题")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("思考中..."):
                result = answer(
                    question,
                    st.session_state.store,
                    llm,
                    title,
                    history=[(t["q"], t["a"]) for t in st.session_state.history],
                )
        except Exception as exc:
            st.error(f"回答失败：{exc}")
            st.stop()
        st.write(result.text)
        if result.pages:
            st.caption("参考页码：" + "、".join(str(p) for p in result.pages))
    st.session_state.history.append({"q": question, "a": result.text, "pages": result.pages})
    st.session_state.history = st.session_state.history[-CHAT_CONTEXT_LENGTH:]

col1, col2 = st.columns(2)
if col1.button("结束对话并总结", use_container_width=True):
    with st.spinner("正在总结对话..."):
        summary = summarize_conversation([(t["q"], t["a"]) for t in st.session_state.history], llm, title)
    st.success(summary)
if col2.button("清空对话", use_container_width=True):
    st.session_state.history = []
    st.rerun()
