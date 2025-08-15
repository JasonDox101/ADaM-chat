import pandas as pd
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import ADaMConfig

class ADaMDataProcessor:
    """ADaM Data Processor for handling PDF documents"""
    
    def __init__(self):
        self.pdf_documents = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ADaMConfig.CHUNK_SIZE,
            chunk_overlap=ADaMConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_data(self) -> None:
        """Load PDF document from configured path"""
        try:
            if not os.path.exists(ADaMConfig.ADAM_PDF_PATH):
                raise FileNotFoundError(f"PDF file not found: {ADaMConfig.ADAM_PDF_PATH}")
            
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(ADaMConfig.ADAM_PDF_PATH)
            self.pdf_documents = loader.load()
            
        except Exception as e:
            raise Exception(f"PDF loading failed: {str(e)}")
    
    def create_documents(self) -> List[Document]:
        """Create LangChain document objects from loaded PDF data"""
        if not self.pdf_documents:
            raise Exception("No PDF data loaded. Please call load_data() first.")
        
        documents = []
        
        # 对每一页进行文本分割
        for page_num, doc in enumerate(self.pdf_documents):
            # 分割文本为更小的块
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for chunk_num, chunk in enumerate(chunks):
                # 创建文档对象，包含丰富的元数据
                metadata = {
                    "source": "ADaMIG_v1.3.pdf",
                    "page": page_num + 1,
                    "chunk": chunk_num + 1,
                    "total_pages": len(self.pdf_documents),
                    "document_type": "ADaM_Implementation_Guide",
                    "chunk_size": len(chunk)
                }
                
                # 添加页面信息到内容中
                content = f"[Page {page_num + 1}] {chunk}"
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        # 移除调试输出
        # print(f"Created {len(documents)} document chunks from PDF")
        return documents

# 删除整个WHIDataProcessor类
class WHIDataProcessor:
    """WHI Data Processor for handling medical research data"""
    
    def __init__(self):
        self.mesa_data = None
        self.dataset_desc = None
    
    def load_data(self) -> None:
        """Load data files from configured paths"""
        try:
            self.mesa_data = pd.read_csv(WHIConfig.MESA_DATA_PATH)
            self.dataset_desc = pd.read_csv(WHIConfig.DATASET_DESC_PATH)
        except Exception as e:
            raise Exception(f"Data loading failed: {str(e)}")
    
    def create_documents(self) -> List[Document]:
        """Create LangChain document objects from loaded data"""
        documents = []
        
        # Process variable-level data
        for _, row in self.mesa_data.iterrows():
            content = f"""Variable Name: {row['Variable name']}
Variable Description: {row['Variable description']}
Variable Type: {row['Type']}
Dataset: {row['Dataset name']}
Study: {row['Study']}
Database: {row['Database']}"""
            
            metadata = {
                "variable_accession": row['Variable accession'],
                "variable_name": row['Variable name'],
                "dataset_accession": row['Dataset accession'],
                "dataset_name": row['Dataset name'],
                "study": row['Study'],
                "type": "variable"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Process dataset-level data
        for _, row in self.dataset_desc.iterrows():
            content = f"""Dataset Name: {row['Dataset name']}
Dataset Description: {row['Dataset description']}
Study: {row['Study']}
Database: {row['Database']}
URL: {row.get('URL', 'N/A')}"""
            
            metadata = {
                "dataset_accession": row['Dataset accession'],
                "dataset_name": row['Dataset name'],
                "study": row['Study'],
                "database": row['Database'],
                "type": "dataset"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents