# 文档问答系统 (RAG)

基于 LangChain + Chroma 向量数据库的文档问答系统，支持 PDF、Word、TXT 格式文档。

## 功能

- 📄 支持 PDF、Word、TXT 文档解析
- 🔍 向量相似度检索
- 🤖 基于大模型的智能问答
- 💾 本地知识库持久化

## 技术栈

- LangChain
- Chroma (向量数据库)
- BGE Embedding (中文向量化)
- Qwen2.5 (SiliconFlow API)

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/DetroySY/rag.git
cd rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 SiliconFlow API Key：

```
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 免费注册: https://siliconflow.cn

### 4. 添加文档

在 `docs` 文件夹中放入要问答的文档（PDF/Word/TXT）：
```
docs/
├── 文档1.pdf
├── 文档2.docx
└── 笔记.txt
```

### 5. 运行

```bash
python main.py
```

首次运行输入 `1` 构建知识库，然后就可以提问了。

## 交互命令

| 命令 | 说明 |
|------|------|
| `add` | 重新添加文档到知识库 |
| `q` | 退出 |

## 项目结构

```
rag/
├── main.py              # 主程序
├── document_loader.py   # 文档解析
├── requirements.txt     # 依赖
├── .env.example         # 环境变量示例
├── docs/                # 文档文件夹（需手动创建）
└── chroma_db/           # 向量数据库（自动生成）
```

