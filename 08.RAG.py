# 08-RAG.py
# 完整的 RAG 示例：使用本地 Ollama 模型 + Chroma 向量库

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # 或者用 ChatOllama
from langchain_classic.chains import RetrievalQA

# ---------- 1. 准备文档：创建 rag指南.txt ----------
DOC_DIR = "rag指南.txt"


def create_doc():
    """如果文档不存在则创建"""
    if not os.path.exists(DOC_DIR):
        content = """
RAG全称是Retrieval-Augmented Generation，由Meta AI在2020年提出。

核心思想：
RAG将信息检索系统与大型语言模型结合。当用户提问时，系统先从知识库中检索相关文档，
然后将这些文档作为上下文一起提供给LLM，让LLM基于这些私有知识生成答案。

RAG的三大优势：
1. 知识实时更新：无需重新训练模型，只需更新知识库即可
2. 减少幻觉：LLM基于真实文档回答，大幅减少编造内容的概率
3. 可溯源：每段回答都能追溯到具体的来源文档

ChromaDB简介：
ChromaDB是一个开源的向量数据库，专为AI应用设计。它支持：
- 多种Embedding模型
- 高效的近似最近邻搜索（ANN）
- 元数据过滤
- 本地持久化存储
"""
        with open(DOC_DIR, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"已创建文档: {DOC_DIR}")
    else:
        print(f"文档已存在: {DOC_DIR}")


# ---------- 2. 加载并分割文档 ----------
def load_and_split():
    loader = TextLoader(DOC_DIR, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 每块最大字符数（可调整）
        chunk_overlap=50,  # 块之间重叠字符数
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    print(f"文档已分割为 {len(docs)} 个文本块")
    return docs


# ---------- 3. 创建 Embedding 模型（使用已下载的 bge-small-zh）----------
def get_embeddings():
    print("正在加载 Embedding 模型...")
    local_model_path = "D:/Learn/code/models/embeddings/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding 模型加载完成")
    return embeddings


# ---------- 4. 创建或加载向量数据库 ----------
PERSIST_DIR = "./chroma_db"  # 向量库持久化目录


def get_vectorstore(embeddings, docs=None):
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # 如果已有向量库，直接加载
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print("加载已有向量数据库")
    else:
        # 否则从文档创建
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print("创建新的向量数据库并持久化")
    return vectorstore


# ---------- 5. 构建检索问答链 ----------
def build_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 检索3个最相关块
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,  # 返回来源，便于溯源
        verbose=True  # 打印检索过程
    )
    return qa_chain


# ---------- 6. 主程序 ----------
if __name__ == "__main__":
    # 创建文档（如果不存在）
    create_doc()

    # 加载并分割文档
    docs = load_and_split()

    # 获取 embedding 模型
    embeddings = get_embeddings()

    # 创建/加载向量库
    vectorstore = get_vectorstore(embeddings, docs)

    # 初始化本地 Ollama 模型（确保服务已启动，模型已下载）
    # 如果你用的是 qwen2.5:1.5b，模型名填写准确
    llm = Ollama(model="qwen2.5:1.5b", temperature=0.7)
    # 如果你希望使用 ChatOllama（消息格式），可以换成：
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)

    # 构建问答链
    qa = build_qa_chain(vectorstore, llm)

    # 测试提问
    question = "什么是RAG？它有什么优势？"
    result = qa.invoke({"query": question})

    print("\n" + "=" * 50)
    print(f"问题: {question}")
    print(f"回答: {result['result']}")
    print("\n参考来源：")
    for i, doc in enumerate(result['source_documents']):
        print(f"{i + 1}. {doc.page_content[:100]}...")