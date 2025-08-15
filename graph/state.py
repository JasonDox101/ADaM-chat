from typing import TypedDict, List, Optional, Dict, Any
from langchain.schema import Document

class ADaMRAGState(TypedDict):
    """ADaM RAG system state management."""
    # User input
    question: str
    
    # Language control
    output_language: Optional[str]  # 输出语言控制
    
    # Conversation history context
    conversation_history: Optional[List[Dict[str, Any]]]  # Historical conversation records
    context_summary: Optional[str]  # Context summary
    related_previous_qa: Optional[List[Dict[str, str]]]  # Related historical Q&A
    
    # Processing workflow
    question_type: Optional[str]  # Question classification result
    retrieved_documents: Optional[List[Document]]  # Retrieved relevant documents
    context_analysis: Optional[Dict[str, Any]]  # Context analysis result
    
    # Final output
    answer: Optional[str]  # Generated answer
    confidence_score: Optional[float]  # Answer confidence score
    source_documents: Optional[List[str]]  # Source document references
    
    # Processing tracking
    processing_steps: Optional[List[str]]  # Processing step records
    error_message: Optional[str]  # Error information
    
    # Error handling
    error: Optional[str]
    
    # 添加缺失的字段
    sources: Optional[List[Dict[str, Any]]]  # 源文档信息
    summary_answer: Optional[str]  # 总结答案
    context: Optional[str]  # 检索到的上下文信息