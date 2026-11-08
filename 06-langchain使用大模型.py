
# 引入核心库
import os
from langchain_openai import ChatOpenAI

# 设置电脑中的环境变量
from dotenv import load_dotenv

# 把相对路径改为绝对路径
current_path=os.path.dirname(__file__)#当前文件所在目录
env_path=os.path.join(current_path,".env")# .env文件路径拼接成绝对路径
load_dotenv(dotenv_path=env_path,override=True)

# 获取电脑中的环境变量
api_key=os.getenv("QWEN_API_KEY")
base_url=os.getenv("QWEN_BASE_URL")
model=os.getenv("QWEN_MODEL")


def load_agent():
    # 创建大模型对象
    llm=ChatOpenAI(model=model,api_key=api_key,base_url=base_url)
    return llm


if __name__ == '__main__':
    # 调用函数加载模型对象
    llm=load_agent()
    # 使用模型对象进行推理
    result=llm.invoke("你是谁？")
    print(result.content)
