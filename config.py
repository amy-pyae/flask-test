import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    UPLOAD_FOLDER = 'uploads'
    SYSTEM_API_KEY = os.getenv("SYSTEM_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_USER_NAME = os.getenv("REDIS_USER_NAME")
    REDIS_API_PASSWORD = os.getenv("REDIS_API_PASSWORD")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")