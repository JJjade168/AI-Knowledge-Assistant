# 引入核心库
import os
from langchain_openai import ChatOpenAI

# 直接指定本地 Ollama 的配置
base_url = "http://127.0.0.1:11434/v1"   # 本机 Ollama 服务地址
api_key = "ollama"                        # 任意字符串
model = "qwen2.5:1.5b"                    # 你下载的模型名

def load_agent():
    # 创建大模型对象
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    return llm

if __name__ == "__main__":
    # 调用函数加载模型对象
    llm = load_agent()
    # 使用模型对象进行推理
    result = llm.invoke("你是谁？")
    print(result.content)