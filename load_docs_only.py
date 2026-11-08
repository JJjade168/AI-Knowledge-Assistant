import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader

# 使用绝对路径避免相对路径问题
DOC_DIR = Path(r"D:\Learn\code\docs")

def load_docs():
    all_docs = []
    # 只使用 TextLoader 加载 .txt 和 .md 文件（.md 也当作纯文本读取）
    for ext in ["*.md", "*.txt"]:
        for file in DOC_DIR.glob(ext):
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {file} 出错: {e}")
    return all_docs

if __name__ == "__main__":
    print("开始加载文档...")
    docs = load_docs()
    if not docs:
        print("错误：没有找到任何文档，请检查 docs 文件夹路径和内容。")
    else:
        # 确保第一个文档内容与老师要求一致
        print(f"有{len(docs)}个Document资料")
        print(f"第一个的内容:{docs[0].page_content[:200]}")
        print(f"第一个的元信息:{docs[0].metadata}")