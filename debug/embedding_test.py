#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author      : wangbo
@Mail        : wangbo@tingyun.com
@Time        : 2025/3/13 18:30
@Version     : 1.0
@Description :
"""

from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from llama_index.core import TreeIndex
from llama_index.core.readers import StringIterableReader
from openai import OpenAI

os.environ.setdefault("OPENAI_API_KEY", "sk-cc2df096be3c4382b3da9a63e5b3b267")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-cc2df096be3c4382b3da9a63e5b3b267")

os.environ.setdefault("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, TextSplitter, TokenTextSplitter


class StringLoader(BaseLoader):
    """自定义 Loader 从字符串加载文档。"""

    def __init__(self, string_content: str):
        self.string_content = string_content

    def load(self):
        yield Document(page_content=self.string_content)

doc = StringLoader("Ketanji Brown Jackson say hello ").load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(doc)


db = Chroma.from_documents(documents, DashScopeEmbeddings())

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)