import os
from dotenv import load_dotenv

# 明确指定 .env 文件的绝对路径（确保正确）
env_path = r"D:\Learn\code\.env"
load_dotenv(dotenv_path=env_path, override=True)

# 读取三个变量
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_BASE_URL")
model = os.getenv("QWEN_MODEL")

print(f"QWEN_API_KEY = {api_key}")
print(f"QWEN_BASE_URL = {base_url}")
print(f"QWEN_MODEL = {model}")