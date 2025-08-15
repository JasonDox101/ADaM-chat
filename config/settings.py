import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class ADaMConfig:  # 重命名为ADaMConfig
    """ADaM RAG system configuration settings."""
    
    # Alibaba Cloud Qwen configuration
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen3-30b-a3b-thinking-2507"
    
    # PDF文档路径配置
    ADAM_PDF_PATH = "./ADaMIG_v1.3.pdf"
    
    # Vector database configuration
    VECTOR_STORE_PATH = "./adam_vectorstore"  # 更新向量存储路径
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # RAG configuration
    RETRIEVAL_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration completeness."""
        if not cls.DASHSCOPE_API_KEY:
            print(f"DASHSCOPE_API_KEY: {cls.DASHSCOPE_API_KEY}")
            print("Please check if .env file exists and contains correct API key")
            return False
        return True
