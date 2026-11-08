# 09-server.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"   # 强制离线，防止联网超时
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# RAG 模块
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# 配置路径
DOC_DIR = Path("./docs")
CHROMA_DIR = Path("./chroma_db")
# 请确保这个路径是正确的本地 embedding 模型路径
EMBEDDING_MODEL_PATH = "D:/Learn/code/models/embeddings/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"

# 全局变量
vectorstore = None
qa_chain = None

def load_documents():
    all_docs = []
    if not DOC_DIR.exists():
        print(f"警告：文档目录 {DOC_DIR} 不存在，请创建并放入文档。")
        return all_docs
    for file in DOC_DIR.glob("*.md"):
        try:
            loader = UnstructuredMarkdownLoader(str(file))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"加载 {file} 失败: {e}")
    for file in DOC_DIR.glob("*.txt"):
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"加载 {file} 失败: {e}")
    print(f"成功加载 {len(all_docs)} 个文档")
    return all_docs

def build_or_load_vectorstore():
    global vectorstore
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("检测到已有向量库，直接加载...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        print("向量库加载完成")
        # 尝试获取文档数量
        try:
            count = vectorstore._collection.count()
            print(f"向量库中文档数量: {count}")
        except:
            print("无法获取文档数量，但向量库对象已创建")
    else:
        print("未找到向量库，开始构建...")
        docs = load_documents()
        if not docs:
            print("错误：没有文档，无法构建向量库。")
            return None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"文档已分割为 {len(chunks)} 个文本块")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        print(f"向量库已创建并保存至 {CHROMA_DIR.absolute()}")
    return vectorstore


def build_qa_chain():
    global qa_chain
    if vectorstore is None:
        print("错误：向量库未初始化")
        return None
    print("开始构建问答链...")
    try:
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        # 加载 .env 文件（确保路径正确）
        load_dotenv()

        api_key = os.getenv("Qwen_API_KEY")
        base_url = os.getenv("Qwen_BASE_URL")
        model_name = os.getenv("Qwen_MODEL")

        print(f"读取配置 - API Key: {api_key[:10]}... Base URL: {base_url}, Model: {model_name}")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7
        )
        print("百炼 API 模型加载成功")
    except Exception as e:
        print(f"加载 API 模型失败: {e}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    print("问答链构建完成")
    return qa_chain

def init_rag():
    print("初始化 RAG 系统...")
    if build_or_load_vectorstore() is not None:
        build_qa_chain()
        print("RAG 系统就绪")
    else:
        print("RAG 系统初始化失败")

# FastAPI 应用
app = FastAPI(title="RAG 问答服务", description="基于本地知识库的智能问答")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ai_chat", response_model=Answer)
def ai_chat(q: Question):
    global qa_chain
    print(f"收到请求，qa_chain 是否为 None: {qa_chain is None}")
    if qa_chain is None:
        # 尝试重新初始化（防止启动时未成功）
        print("尝试重新初始化 RAG...")
        init_rag()
    if qa_chain is None:
        return Answer(answer="RAG 系统尚未初始化，请检查服务器日志。", sources=[])
    try:
        result = qa_chain.invoke({"query": q.question})
        answer = result["result"]
        sources = []
        for doc in result.get("source_documents", []):
            source = Path(doc.metadata.get("source", "unknown")).name
            sources.append(source)
        return Answer(answer=answer, sources=sources)
    except Exception as e:
        print(f"处理请求时出错: {e}")
        return Answer(answer=f"处理出错：{str(e)}", sources=[])

if __name__ == "__main__":
    init_rag()
    uvicorn.run(app="09-server:app", host="0.0.0.0", port=8000, reload=True)