from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from graph.state import ADaMRAGState  # 需要更新state文件
from llm.qwen_client import QwenLLMClient
from vector_store.manager import ADaMVectorStoreManager  # 更新导入
from data.processor import ADaMDataProcessor  # 更新导入
from config.settings import ADaMConfig  # 更新导入
import json
import re
from datetime import datetime

class ADaMRAGSystem:  # 重命名类
    """Core ADaM RAG system for document analysis and question answering."""
    
    def __init__(self):
        self.llm_client = QwenLLMClient()
        self.vector_manager = ADaMVectorStoreManager()
        self.data_processor = ADaMDataProcessor()
        self.workflow = None
        self.conversation_memory = []  # Conversation memory storage
        self.max_history_length = 10  # Maximum number of historical conversations to keep
        self._initialize_system()
        self._build_workflow()
    
    def _initialize_system(self) -> None:
        """Initialize the RAG system components."""
        try:
            # Load PDF data
            self.data_processor.load_data()
            
            # Try to load existing vector store
            if not self.vector_manager.load_vector_store():
                documents = self.data_processor.create_documents()
                self.vector_manager.create_vector_store(documents)

                
        except Exception as e:
            raise Exception(f"System initialization failed: {str(e)}")
    
    def process_question(self, question: str, conversation_history: List[Dict] = None, output_language: str = "english") -> Dict[str, Any]:
        """Process question using enhanced RAG workflow."""
        try:
            if conversation_history is None:
                conversation_history = []
            
            # Initialize state
            initial_state = {
                "question": question,
                "conversation_history": conversation_history,
                "output_language": output_language,
                "processing_steps": []
            }
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            # Ensure all required keys are present in the final result
            final_result = {
                "question": question,
                "output_language": output_language,
                "conversation_history": conversation_history,
                "context_summary": result.get("context_summary", ""),
                "related_previous_qa": result.get("related_previous_qa", []),
                "question_type": result.get("question_type", "general"),
                "retrieved_documents": result.get("retrieved_documents", []),
                "answer": result.get("answer", "No answer generated"),
                "summary_answer": result.get("summary_answer", "No summary generated"),
                "confidence_score": result.get("confidence_score", 0.0),
                "sources": result.get("sources", []),
                "processing_steps": result.get("processing_steps", [])
            }
            
            # Save to conversation memory
            self._save_to_memory(question, final_result)
            
            return final_result
        except Exception as e:
            return {
                "error": f"Question processing failed: {str(e)}",
                "answer": "Sorry, an error occurred while processing your question.",
                "summary_answer": "The system is temporarily unable to process your question. Please try again later.",
                "confidence_score": 0.0,
                "sources": [],
                "processing_steps": [f"Error: {str(e)}"]
            }
    
    def _save_to_memory(self, question: str, result: Dict[str, Any]):
        """Save conversation to memory."""
        try:
            qa_pair = {
                "question": question,
                "answer": result.get("summary_answer", ""),
                "detailed_answer": result.get("answer", ""),
                "timestamp": datetime.now().isoformat(),
                "confidence": result.get("confidence_score", 0),
                "question_type": result.get("question_type", "general")
            }
            
            self.conversation_memory.append(qa_pair)
            
            # Maintain memory length limit
            if len(self.conversation_memory) > self.max_history_length:
                self.conversation_memory.pop(0)
        except Exception as e:
            pass
    
    def _build_workflow(self) -> None:
        """构建优化的LangGraph工作流程 - 减少LLM调用次数"""
        workflow = StateGraph(ADaMRAGState)
        
        # 添加优化后的节点
        workflow.add_node("analyze_context_and_classify", self._analyze_context_and_classify)  # 合并节点1
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_answer_with_summary", self._generate_answer_with_summary)  # 合并节点2
        workflow.add_node("validate_answer", self._validate_answer)
        
        # 设置优化后的边 - 新工作流程
        workflow.set_entry_point("analyze_context_and_classify")
        workflow.add_edge("analyze_context_and_classify", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer_with_summary")
        workflow.add_edge("generate_answer_with_summary", "validate_answer")
        workflow.add_edge("validate_answer", END)
        
        self.workflow = workflow.compile()
    
    def _analyze_context_and_classify(self, state: ADaMRAGState) -> Dict[str, Any]:
        """合并的上下文分析和问题分类节点 - 优化为一次LLM调用"""
        try:
            question = state["question"]
            history = state.get("conversation_history", [])
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting combined context analysis and question classification")
            
            # 构建合并的提示词
            combined_prompt = f"""
你是一位专业的ADaM（Analysis Data Model）临床数据分析专家。请同时完成以下两个任务：

**任务1：上下文分析**
当前问题：{question}

历史对话：
{self._format_history_for_analysis(history) if history else "无历史对话"}

请分析当前问题与历史对话的关系。

**任务2：问题分类**
请将问题分类到以下类别之一：
- "adsl": ADSL主题级分析数据集相关问题
- "bds": BDS基础数据结构相关问题  
- "occds": OCCDS事件数据结构相关问题
- "variable": 变量相关问题（PARAMCD, AVAL, BASE等）
- "methodology": 方法论相关问题（编程、验证、溯源性等）
- "general": ADaM标准与概念相关的一般性问题

请以JSON格式返回结果：
{{
    "context_analysis": {{
        "is_related": true/false,
        "context_summary": "上下文摘要",
        "related_qa_indices": [0, 1],
        "reasoning": "关系分析推理"
    }},
    "question_type": "分类结果"
}}
"""
            
            messages = [
                {"role": "system", "content": "你是专业的临床数据分析助手，擅长分析对话上下文关系和问题分类。请严格按照JSON格式返回结果。"},
                {"role": "user", "content": combined_prompt}
            ]
            
            try:
                result = self.llm_client.generate_response(messages)
                
                # 清理可能的markdown格式
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].strip()
                
                combined_result = json.loads(result)
                context_analysis = combined_result.get("context_analysis", {})
                question_type = combined_result.get("question_type", "general")
                
            except Exception as e:
                # 如果LLM分析失败，使用增强的关键词匹配
                context_analysis = self._enhanced_context_analysis(question, history)
                question_type = "general"
            
            # 提取相关的历史问答
            related_qa = []
            if context_analysis.get("is_related", False):
                for idx in context_analysis.get("related_qa_indices", []):
                    if 0 <= idx < len(history):
                        related_qa.append({
                            "question": history[idx]["question"],
                            "answer": history[idx]["answer"]
                        })
            
            # 确保分类结果有效
            valid_types = ["adsl", "bds", "occds", "variable", "methodology", "general"]
            if question_type not in valid_types:
                question_type = "general"
            
            processing_steps.append(f"Combined analysis completed - Type: {question_type}, Context related: {context_analysis.get('is_related', False)}")
            
            return {
                "context_summary": context_analysis.get("context_summary", ""),
                "related_previous_qa": related_qa,
                "is_context_related": context_analysis.get("is_related", False),
                "question_type": question_type,
                "processing_steps": processing_steps
            }
            
        except Exception as e:
            return {
                "context_summary": "Combined analysis failed",
                "related_previous_qa": [],
                "is_context_related": False,
                "question_type": "general",
                "error": f"Combined analysis failed: {str(e)}",
                "processing_steps": processing_steps + [f"Combined analysis failed: {str(e)}"]
            }
    
    def _enhanced_context_analysis(self, question: str, history: List[Dict]) -> Dict[str, Any]:
        """Enhanced context analysis using keyword matching and semantic similarity."""
        question_lower = question.lower()
        related_indices = []
        
        # Medical term keywords for ADaM research
        medical_keywords = ['paramcd', 'param', 'aval', 'base', 'chg', 'ablfl', 'baseline', 'analysis', 'visit']
        dataset_keywords = ['adam', 'adsl', 'adae', 'adlb', 'advs', 'dataset', 'study', 'variable', 'cdisc']
        
        for i, item in enumerate(history):
            hist_question = item.get("question", "").lower()
            hist_answer = item.get("answer", "").lower()
            
            # Keyword overlap detection
            question_words = set(question_lower.split())
            hist_words = set(hist_question.split())
            overlap = len(question_words & hist_words)
            
            # Medical term matching
            medical_overlap = any(term in question_lower and term in hist_question for term in medical_keywords)
            dataset_overlap = any(term in question_lower and term in hist_question for term in dataset_keywords)
            
            if overlap > 1 or medical_overlap or dataset_overlap:
                related_indices.append(i)
        
        return {
            "is_related": len(related_indices) > 0,
            "context_summary": f"Found {len(related_indices)} related historical conversations" if related_indices else "No related historical conversations",
            "related_qa_indices": related_indices,
            "reasoning": "Enhanced analysis based on keyword matching and ADaM terminology recognition"
        }
    
    def _classify_question(self, state: ADaMRAGState) -> Dict[str, Any]:  # Updated parameter type
        """Question classification node."""
        try:
            question = state["question"]
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting question classification")
            
            # 当前使用英文，建议保持不变（内部处理逻辑）
            classification_prompt = f"""
            Please classify the following question about ADaM (Analysis Data Model) data:
            
            Question: {question}
            
            Classify into one of these categories:
            - "adsl": ADSL主题级分析数据集相关问题
            - "bds": BDS基础数据结构相关问题  
            - "occds": OCCDS事件数据结构相关问题
            - "adtte": ADTTE生存分析数据集相关问题
            - "safety": 安全性数据集（ADAE, ADCM等）相关问题
            - "efficacy": 疗效数据集（ADEFF, ADRS等）相关问题
            - "laboratory": 实验室数据集（ADLB, ADVS等）相关问题
            - "pkpd": PK/PD数据集（ADPC, ADPP等）相关问题
            - "variable": 变量相关问题（PARAMCD, AVAL, BASE等）
            - "methodology": 方法论相关问题（编程、验证、溯源性等）
            - "general": ADaM标准与概念相关的一般性问题
            
            Return only the category name.
            """
            
            messages = [
                {"role": "system", "content": "You are a professional clinical data analysis assistant."},
                {"role": "user", "content": classification_prompt}
            ]
            
            question_type = self.llm_client.generate_response(messages).strip().lower()
            
            # Ensure classification result is valid
            if question_type not in ["variable", "dataset", "general"]:
                question_type = "general"
            
            processing_steps.append(f"Question classification completed: {question_type}")
            
            return {
                "question_type": question_type,
                "processing_steps": processing_steps
            }
        except Exception as e:
            return {
                "question_type": "general",
                "error": f"Question classification failed: {str(e)}",
                "processing_steps": processing_steps + [f"Question classification failed: {str(e)}"]
            }
    
    def _retrieve_documents(self, state: ADaMRAGState) -> Dict[str, Any]:  # Updated parameter type
        """Document retrieval node."""
        try:
            question = state["question"]
            question_type = state.get("question_type", "general")
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting document retrieval")
            
            # Generate optimized search query
            search_query = self._generate_search_query(question, question_type)
            processing_steps.append(f"Generated search query: {search_query}")
            
            # Execute similarity search
            retrieved_docs = self.vector_manager.similarity_search(
                search_query, 
                k=ADaMConfig.RETRIEVAL_K  # Updated to ADaMConfig
            )
            
            processing_steps.append(f"Retrieved {len(retrieved_docs)} relevant documents")
            
            return {
                "search_query": search_query,
                "retrieved_documents": retrieved_docs,
                "processing_steps": processing_steps
            }
        except Exception as e:
            return {
                "error": f"Document retrieval failed: {str(e)}",
                "retrieved_documents": [],
                "processing_steps": processing_steps + [f"Document retrieval failed: {str(e)}"]
            }
    
    def _generate_search_query(self, question: str, question_type: str) -> str:
        """Generate optimized search query."""
        try:
            # 当前使用英文，建议保持不变（内部处理逻辑）
            query_prompt = f"""
            Based on question type "{question_type}" and user question, generate optimized keyword queries suitable for retrieval in ADaM clinical data.
            
            User question: {question}
            
            **检索策略：**
            - 提取核心的ADaM术语、变量名或数据集名
            - 包含相关的CDISC标准术语
            - 考虑同义词和相关概念
            - 优化检索精度和召回率
            
            **ADaM核心术语库：**
            - 数据集：ADSL, ADAE, ADCM, ADLB, ADVS, ADEFF, ADRS, ADTTE, ADPC, ADPP
            - 关键变量：USUBJID, PARAMCD, PARAM, AVAL, BASE, CHG, PCHG, ABLFL, ANL01FL
            - 结构类型：BDS, OCCDS, ADSL
            - 分析概念：baseline, endpoint, safety, efficacy, PK, PD, survival
            
            Please generate concise but comprehensive search queries. Return only the query terms, without any other content.
            """
            
            messages = [
                {"role": "system", "content": "You are a professional clinical research assistant."},
                {"role": "user", "content": query_prompt}
            ]
            
            return self.llm_client.generate_response(messages).strip()
        except:
            # If generation fails, return original question
            return question
    
    def _generate_answer_with_summary(self, state: ADaMRAGState) -> Dict[str, Any]:
        """合并的答案生成和摘要节点 - 优化为一次LLM调用"""
        try:
            question = state["question"]
            retrieved_docs = state.get("retrieved_documents", [])
            related_qa = state.get("related_previous_qa", [])
            context_summary = state.get("context_summary", "")
            output_language = state.get("output_language", "english")
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting combined answer generation and summarization")
            
            # 构建文档上下文
            context = self._build_context(retrieved_docs)
            
            # 构建上下文信息
            context_info = ""
            if related_qa:
                context_info = "\n\n**相关历史对话：**\n"
                for i, qa in enumerate(related_qa, 1):
                    context_info += f"{i}. Q: {qa['question']}\n   A: {qa['answer'][:200]}...\n\n"
            
            # 根据语言设置不同的指令
            if output_language == "chinese":
                language_instruction = "请用中文回答。确保所有回答内容都使用中文，包括临床术语的中文表达。"
                system_content = """你是一位专业的ADaM（Analysis Data Model）临床数据分析专家助手。请同时提供详细回答和结构化摘要。"""
                
                combined_prompt = f"""
{language_instruction}

请基于提供的文档上下文和对话历史回答用户问题，并同时提供详细回答和结构化摘要。

用户当前问题：{question}

文档上下文：
{context}
{context_info}

请以JSON格式返回结果，包含以下两部分：
{{
    "detailed_answer": "详细的专业回答（严格遵循markdown格式）",
    "summary_answer": "结构化摘要（200-400字，包含核心概念、关键技术要点、实用指导、ADaM标准关联、业务价值）"
}}

**格式要求：**
1. **详细回答**：使用## 主标题，### 副标题，- 无序列表，**粗体**强调重要信息
2. **摘要**：简洁但信息丰富，突出关键技术细节和实用指导
3. 使用专业的ADaM/CDISC术语
4. 如果与之前问题相关，请在回答中体现这种关系
"""
            else:
                language_instruction = "Please respond in English. Ensure all content is in English, including clinical terminology."
                system_content = "You are a professional ADaM clinical data analysis assistant. Please provide both detailed answer and structured summary simultaneously."
                
                combined_prompt = f"""
{language_instruction}

Please answer the user's question based on the provided document context and conversation history, and provide both detailed answer and structured summary.

User's current question: {question}

Document context:
{context}
{context_info}

Please return results in JSON format with the following two parts:
{{
    "detailed_answer": "Detailed professional answer (strictly follow markdown format)",
    "summary_answer": "Structured summary (200-400 words, including core concepts, key technical points, practical guidance, ADaM standard connections, business value)"
}}

**Format requirements:**
1. **Detailed answer**: Use ## main titles, ### subtitles, - unordered lists, **bold** for emphasis
2. **Summary**: Concise but information-rich, highlighting key technical details and practical guidance
3. Use professional ADaM/CDISC terminology
4. If related to previous questions, reflect this relationship in the answer
"""
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_prompt}
            ]
            
            try:
                result = self.llm_client.generate_response(messages)
                
                # 清理可能的markdown格式
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].strip()
                
                combined_result = json.loads(result)
                detailed_answer = combined_result.get("detailed_answer", "")
                summary_answer = combined_result.get("summary_answer", "")
                
            except Exception as e:
                # 如果解析失败，尝试直接使用结果作为详细回答
                detailed_answer = result if 'result' in locals() else "Answer generation failed"
                summary_answer = "Summary generation failed"
            
            # 应用markdown格式标准化
            detailed_answer = self._ensure_markdown_format(detailed_answer)
            
            # 提取源信息
            sources = self._extract_sources(retrieved_docs)
            
            # 确保摘要不为空
            if not summary_answer or summary_answer.strip() == "":
                summary_answer = "Combined generation completed but summary is empty"
            
            processing_steps.append("Combined answer generation and summarization completed")
            
            return {
                "context": context,
                "answer": detailed_answer,
                "summary_answer": summary_answer,
                "sources": sources,
                "processing_steps": processing_steps
            }
            
        except Exception as e:
            return {
                "error": f"Combined generation failed: {str(e)}",
                "answer": "Sorry, an error occurred while generating the answer.",
                "summary_answer": "Summary generation failed",
                "sources": [],
                "processing_steps": processing_steps + [f"Combined generation failed: {str(e)}"]
            }
    
    def _build_context(self, documents: List) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Information {i}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            # 适配ADaM PDF文档的元数据结构
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "document_type": doc.metadata.get("document_type", "ADaM_Implementation_Guide"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk": doc.metadata.get("chunk", "N/A"),
                "total_pages": doc.metadata.get("total_pages", "N/A")
            }
            sources.append(source_info)
        return sources
    
    def _summarize_answer(self, state: ADaMRAGState) -> Dict[str, Any]:
        """Answer summarization node - second LLM call."""
        try:
            detailed_answer = state.get("answer", "")
            question = state["question"]
            output_language = state.get("output_language", "english")
            sources = state.get("sources", [])
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting answer summarization")
            
            if not detailed_answer:
                fallback_summary = "No detailed answer available for summarization"
                return {
                    "summary_answer": fallback_summary,
                    "sources": sources,
                    "processing_steps": processing_steps + ["Detailed answer is empty, cannot summarize"]
                }
            
            # 根据语言设置不同的总结指令
            if output_language == "chinese":
                summary_instruction = "请用中文提供结构化的详细总结回答。"
                system_content = "你是专业的ADaM临床数据分析助手。请提供结构化、信息丰富但简洁的中文总结，确保包含关键技术细节和实用指导。"
                
                summary_prompt = f"""
{summary_instruction}

基于以下详细回答，请提供一个结构化的总结（200-400字），包含以下要素：

详细回答：
{detailed_answer}

请确保总结包含：
1. **核心概念/定义**：简明扼要地说明主要概念或问题的本质
2. **关键技术要点**：列出2-3个重要的技术细节、变量、或实施要求
3. **实用指导**：提供具体的操作建议、最佳实践或注意事项
4. **ADaM标准关联**：如适用，说明与CDISC ADaM标准的关系
5. **业务价值**：简述对临床数据分析或法规合规的意义

格式要求：
- 使用专业的ADaM/CDISC术语
- 保持信息密度高但易于理解
- 突出关键数值、变量名或标准要求
- 确保准确性和完整性

只返回总结内容，不要额外的格式说明。
"""
            else:
                summary_instruction = "Please provide a structured detailed summary answer in English."
                system_content = "You are a professional ADaM clinical data analysis assistant. Please provide a structured, information-rich but concise English summary, ensuring key technical details and practical guidance are included."
                
                summary_prompt = f"""
{summary_instruction}

Based on the detailed answer below, please provide a structured summary (200-400 words) that includes the following elements:

Detailed answer:
{detailed_answer}

Please ensure the summary contains:
1. **Core Concept/Definition**: Clearly explain the main concept or essence of the question
2. **Key Technical Points**: List 2-3 most important technical details, variables, or implementation requirements
3. **Practical Guidance**: Provide specific operational advice, best practices, or considerations
4. **ADaM Standard Connection**: If applicable, explain the relationship with CDISC ADaM standards
5. **Business Value**: Briefly describe the significance for clinical data analysis or regulatory compliance

Format requirements:
- Use professional ADaM/CDISC terminology
- Maintain high information density while being understandable
- Highlight key values, variable names, or standard requirements
- Ensure accuracy and completeness

Return only the summary content without additional formatting instructions.
"""
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary_answer = self.llm_client.generate_response(messages)
            
            # 确保summary不为空
            if not summary_answer or summary_answer.strip() == "":
                summary_answer = "Summary generation completed but result is empty"
            
            processing_steps.append("Answer summarization completed")
            
            return {
                "summary_answer": summary_answer,
                "sources": sources,
                "processing_steps": processing_steps
            }
        except Exception as e:
            error_summary = f"Summary generation failed: {str(e)}"
            return {
                "summary_answer": error_summary,
                "sources": state.get("sources", []),
                "error": f"Answer summarization failed: {str(e)}",
                "processing_steps": processing_steps + [f"Answer summarization failed: {str(e)}"]
            }
    
    def _validate_answer(self, state: ADaMRAGState) -> Dict[str, Any]:
        """Answer validation and confidence scoring node."""
        try:
            answer = state.get("answer", "")
            sources = state.get("sources", [])  # 获取sources
            processing_steps = state.get("processing_steps", [])
            processing_steps.append("Starting answer validation")
            
            # Simple confidence assessment
            confidence_score = self._calculate_confidence(answer, sources)
            
            processing_steps.append(f"Answer validation completed, confidence: {confidence_score:.2f}")
            
            return {
                "confidence_score": confidence_score,
                "sources": sources,  # 传递sources
                "processing_steps": processing_steps
            }
        except Exception as e:
            return {
                "confidence_score": 0.0,
                "sources": [],  # 错误时也要返回空的sources
                "error": f"Answer validation failed: {str(e)}",
                "processing_steps": processing_steps + [f"Answer validation failed: {str(e)}"]
            }
    
    def _calculate_confidence(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on answer quality and source availability."""
        # 基础分数，即使没有sources也不应该是0
        base_score = 0.3
        
        if not answer:
            return 0.0
        
        # 答案长度评分 (0.0-0.4)
        answer_length_score = min(len(answer) / 500, 0.4)
        
        # 来源数量评分 (0.0-0.3)
        source_count_score = 0.0
        if sources:
            source_count_score = min(len(sources) / 5, 0.3)
        
        # 综合评分
        confidence = base_score + answer_length_score + source_count_score
        return round(min(confidence, 1.0), 2)
    
    def _format_history_for_analysis(self, history: List[Dict]) -> str:
        """Format conversation history for analysis."""
        formatted = ""
        for i, item in enumerate(history[-5:], 1):  # Only take the last 5 entries
            formatted += f"{i}. Q: {item.get('question', '')}\n   A: {item.get('answer', '')}\n\n"
        return formatted
    
    def _ensure_markdown_format(self, answer: str) -> str:
        """Ensure answer follows standard markdown format without changing content."""
        import re
        
        # Standardize title format
        answer = re.sub(r'^#{1,6}\s*(.+)$', lambda m: f"## {m.group(1).strip()}", answer, flags=re.MULTILINE)
        
        # Standardize list format
        answer = re.sub(r'^[•·*]\s*', '- ', answer, flags=re.MULTILINE)
        
        # Ensure paragraph spacing
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Ensure numerical values are bolded (if not already)
        answer = re.sub(r'(?<!\*)\b(\d+(?:\.\d+)?\s*(?:g/dL|mg/dL|mmHg|%|years|cases|people|items))(?!\*)', r'**\1**', answer)
        
        return answer.strip()