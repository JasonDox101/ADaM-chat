from openai import OpenAI
from typing import List, Dict, Any
from config.settings import ADaMConfig

class QwenLLMClient:
    """Qwen LLM client wrapper for ADaM data analysis."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=ADaMConfig.DASHSCOPE_API_KEY,
            base_url=ADaMConfig.DASHSCOPE_BASE_URL
        )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from LLM."""
        try:
            completion = self.client.chat.completions.create(
                model=ADaMConfig.MODEL_NAME,
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def generate_embedding_query(self, text: str) -> str:
        """Generate optimized query for retrieval."""
        messages = [
            {"role": "system", "content": "You are a professional ADaM clinical data analysis assistant, skilled at converting user questions into precise retrieval queries."},
            {"role": "user", "content": f"Please convert the following question into keyword queries suitable for retrieval in ADaM clinical data: {text}"}
        ]
        return self.generate_response(messages)