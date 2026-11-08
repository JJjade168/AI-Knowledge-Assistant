import os
from dotenv import load_dotenv

#设置环境变量
os.environ["TEST2026"]="123"
os.environ["QWEN_API_KEY"]="abc123"

#获取环境变量
x=os.getenv("TEST2026")
print(os.getenv("QWEN_API_KEY"))
print(x)
