"""
文档问答系统 - RAG 实战项目
"""
import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer, 'strict')


def get_resource_path(relative_path):
    """获取资源文件路径"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# 加载 .env
env_path = get_resource_path('.env')
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path)


from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

from document_loader import DocumentLoader


# Embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHROMA_DB_PATH = get_resource_path("chroma_db")


class DocumentQASystem:
    def __init__(self):
        print("初始化文档问答系统...")

        # 初始化 Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

        # 初始化大模型
        self.llm = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
        )

        # 文档加载器
        self.loader = DocumentLoader()

        # 向量数据库
        self.vectorstore = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """加载已有向量库"""
        if os.path.exists(CHROMA_DB_PATH):
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )
            count = self.vectorstore._collection.count()
            print(f"已加载知识库: {count} 个文档片段")
        else:
            print("知识库为空")

    def add_documents(self, folder_path="docs"):
        """添加文档到知识库"""
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            print("请创建 docs 文件夹并放入 PDF/Word/TXT 文件")
            return False

        docs = self.loader.load_folder(folder_path)
        if not docs:
            print("未找到任何文档")
            return False

        # 存入向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"已添加 {len(docs)} 个文档片段到知识库")
        return True

    def ask(self, question: str) -> str:
        """问答"""
        if self.vectorstore is None:
            return "知识库为空，请先添加文档"

        # 检索
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 构建 RAG 链
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # RAG prompt
        prompt = """你是一个专业的文档问答助手。请根据以下参考文档回答用户的问题。

如果文档中没有相关信息，请如实告知用户。

参考文档:
{context}

用户问题: {question}

回答:"""

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        # 调用
        result = rag_chain.invoke(question)
        return result.content


def main():
    qa = DocumentQASystem()

    # 检查是否有文档，没有则提示添加
    if qa.vectorstore is None or qa.vectorstore._collection.count() == 0:
        print("\n知识库为空，是否添加文档？")
        print("1. 添加 docs 文件夹中的文档")
        print("2. 跳过，直接提问")
        choice = input("选择 (1/2): ").strip()

        if choice == "1":
            qa.add_documents("docs")

    print("\n=== 文档问答系统 ===")
    print("支持：PDF、Word、TXT 文档问答")
    print("命令: add - 添加文档, q - 退出\n")

    while True:
        query = input("你: ").strip()

        if query.lower() == "q":
            break

        if query.lower() == "add":
            qa.add_documents("docs")
            continue

        if not query:
            continue

        response = qa.ask(query)
        print(f"\nAI: {response}\n")


if __name__ == "__main__":
    main()