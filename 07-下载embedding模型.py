import os
# 设置 Hugging Face 镜像（注意变量名是 HF_ENDPOINT，且使用 https）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings

print("开始下载embedding模型")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_folder="./models/embeddings",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("下载embedding模型完成")