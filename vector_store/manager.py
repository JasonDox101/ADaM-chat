from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
from config.settings import ADaMConfig  # 更新导入

class ADaMVectorStoreManager:  # 重命名类
    """ADaM vector store manager for document retrieval."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create vector store from documents."""
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        self.save_vector_store()
    
    def load_vector_store(self) -> bool:
        """Load existing vector store."""
        try:
            self.vector_store = FAISS.load_local(
                ADaMConfig.VECTOR_STORE_PATH,  # 更新配置引用
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except:
            return False
    
    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(ADaMConfig.VECTOR_STORE_PATH)  # 更新配置引用
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search."""
        if not self.vector_store:
            raise Exception("Vector store not initialized")
        
        k = k or ADaMConfig.RETRIEVAL_K  # 更新配置引用
        return self.vector_store.similarity_search(query, k=k)