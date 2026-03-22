"""
文档加载器 - 支持 PDF、Word、TXT
"""
import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_file(self, file_path: str) -> List:
        """加载单个文件"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

        docs = loader.load()
        return docs

    def load_and_split(self, file_path: str) -> List:
        """加载文件并分块"""
        docs = self.load_file(file_path)
        splits = self.splitter.split_documents(docs)
        print(f"文档已切片: {len(splits)} 个片段")
        return splits

    def load_folder(self, folder_path: str) -> List:
        """加载文件夹下所有支持的文件"""
        all_splits = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    splits = self.load_and_split(file_path)
                    all_splits.extend(splits)
                except Exception as e:
                    print(f"跳过 {file}: {e}")
        return all_splits


if __name__ == "__main__":
    # 测试
    loader = DocumentLoader()

    # 示例：加载当前目录下的 docs 文件夹
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        docs = loader.load_folder(docs_folder)
        print(f"共加载 {len(docs)} 个文档片段")
    else:
        print("请创建 docs 文件夹并放入文档")