import os

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from llama_index.core import TreeIndex
from llama_index.core.readers import StringIterableReader

os.environ.setdefault("DASHSCOPE_API_KEY", "sk-cc2df096be3c4382b3da9a63e5b3b267")

class StringLoader(BaseLoader):
    """自定义 Loader 从字符串加载文档。"""

    def __init__(self, string_content: str):
        self.string_content = string_content

    def load(self):
        yield Document(page_content=self.string_content)

with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 File Q&A with Anthropic")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()

    doc = StringLoader(article).load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc)

    db = Chroma.from_documents(documents, DashScopeEmbeddings())

    docs = db.similarity_search(question)

    st.write(docs[0].page_content)
