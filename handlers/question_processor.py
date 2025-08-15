import markdown
import re
from .utils import StyleConstants

class QuestionProcessor:
    """Question processor class for handling user queries"""
    
    def __init__(self, rag_system=None, system_ready=False):
        self.rag_system = rag_system
        self.system_ready = system_ready
    
    @staticmethod
    def standardize_detailed_answer_format(raw_answer: str) -> str:
        """Standardize the display format of detailed answers in the right panel"""
        # Ensure answer starts with standard structure
        if not raw_answer.strip().startswith('#'):
            raw_answer = f"## Detailed Analysis\n\n{raw_answer}"
        
        # Standardize title format
        raw_answer = re.sub(r'^#{1,6}\s*(.+)$', lambda m: f"## {m.group(1).strip()}", raw_answer, flags=re.MULTILINE)
        
        # Standardize list format
        raw_answer = re.sub(r'^[•·*-]\s*', '- ', raw_answer, flags=re.MULTILINE)
        
        # Ensure proper paragraph spacing
        raw_answer = re.sub(r'\n{3,}', '\n\n', raw_answer)
        
        # 改进的数值范围显示格式 - 更全面的匹配
        # 1. 基本数值+单位格式
        raw_answer = re.sub(r'(\d+(?:\.\d+)?(?:\s*[–—-]\s*\d+(?:\.\d+)?)?\s*(?:g/dL|mg/dL|mmHg|%|years|yrs|IU/L|μg/L|ng/mL|毫克|克))', r'**\1**', raw_answer)
        
        # 2. 置信区间格式
        raw_answer = re.sub(r'(\d+%\s*CI:\s*\d+(?:\.\d+)?[–—-]\d+(?:\.\d+)?)', r'**\1**', raw_answer)
        
        # 3. 风险比和比值比格式
        raw_answer = re.sub(r'((?:HR|RR|OR)\s*=\s*\d+(?:\.\d+)?)', r'**\1**', raw_answer)
        
        # 4. 独立的百分比
        raw_answer = re.sub(r'(?<!\*)\b(\d+(?:\.\d+)?%)(?!\*)', r'**\1**', raw_answer)
        
        # 年龄范围（中文）
        raw_answer = re.sub(r'(\d+[–—-]\d+岁)', r'**\1**', raw_answer)
        
        # 6. 样本量格式
        raw_answer = re.sub(r'(n\s*=\s*\d+(?:,\d+)*)', r'**\1**', raw_answer)
        
        # 7. 相对风险降低格式
        raw_answer = re.sub(r'(降低\d+(?:\.\d+)?%)', r'**\1**', raw_answer)
        
        return raw_answer.strip()
    
    @staticmethod
    def format_summary_answer(raw_summary: str, confidence: float = 0, sources: list = None) -> str:
        """Format summary answer with complete metadata for left panel display"""
        if sources is None:
            sources = []
            
        # Apply standardization
        formatted_summary = QuestionProcessor.standardize_detailed_answer_format(raw_summary)
        
        # Clean up markdown formatting for better plain text display
        formatted_summary = re.sub(r'^#{1,6}\s*(.+)$', r'📋 \1\n', formatted_summary, flags=re.MULTILINE)
        formatted_summary = re.sub(r'^-\s*', '• ', formatted_summary, flags=re.MULTILINE)
        formatted_summary = re.sub(r'\*\*(.+?)\*\*', r'\1', formatted_summary)
        
        # Clean up line breaks
        lines = formatted_summary.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('📋') and cleaned_lines:
                cleaned_lines.append('')
            if line.startswith('• ') and cleaned_lines and not cleaned_lines[-1].startswith('• '):
                if cleaned_lines[-1] != '':
                    cleaned_lines.append('')
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)

        
        return result.strip()

    async def process_question(self, question: str, chat_messages: list, output_language: str = "english"):
        """Main logic for processing questions with language control"""
        try:
            if self.rag_system and self.system_ready:
                # Get conversation history
                conversation_history = []
                
                # Improved conversation history conversion logic
                for i in range(0, len(chat_messages), 2):  # Every two messages form a pair
                    if i < len(chat_messages) and chat_messages[i]['type'] == 'user':
                        user_question = chat_messages[i]['content']
                        assistant_answer = ""
                        
                        # Find corresponding assistant reply
                        if i + 1 < len(chat_messages) and chat_messages[i + 1]['type'] == 'assistant':
                            assistant_answer = chat_messages[i + 1]['content']
                        
                        # Only add complete Q&A pairs to history
                        if assistant_answer:
                            conversation_history.append({
                                "question": user_question,
                                "answer": assistant_answer,
                                "timestamp": chat_messages[i]['timestamp'].isoformat() if hasattr(chat_messages[i]['timestamp'], 'isoformat') else str(chat_messages[i]['timestamp'])
                            })
                        
                # Limit conversation history length to avoid accumulating too much content
                if len(conversation_history) > 3:  # Keep only the most recent 3 complete conversations
                    conversation_history = conversation_history[-3:]
                
                # Process using RAG system with language parameter
                result = self.rag_system.process_question(question, conversation_history, output_language)
                
                # Get detailed answer and summary answer
                detailed_answer = result.get('answer', 'No answer generated')
                
                # Add format standardization
                detailed_answer = self.standardize_detailed_answer_format(detailed_answer)
                
                summary_answer = result.get('summary_answer', 'No summary generated')
                
                # 如果summary_answer为空或None，尝试从detailed_answer生成简短摘要
                if not summary_answer or summary_answer == 'No summary generated':
                    # 生成简短摘要作为备选方案
                    summary_answer = self._generate_fallback_summary(detailed_answer)
                    # 移除调试输出
                    # print(f"Generated fallback summary: {summary_answer}")
                confidence = result.get('confidence_score', 0)
                sources = result.get('sources', [])
                
                # Convert detailed answer to markdown format
                markdown_answer = markdown.markdown(detailed_answer, extensions=['extra', 'codehilite', 'tables', 'toc'])
                
                # Format detailed answer
                formatted_detailed_answer = self._format_detailed_answer(markdown_answer, confidence, sources)
                
                # 在返回结果前格式化summary_answer，包含完整元数据
                formatted_summary = self.format_summary_answer(summary_answer, confidence, sources)
                
                return {
                    'summary_answer': formatted_summary,  # 返回格式化的HTML
                    'detailed_answer': formatted_detailed_answer
                }
            else:
                # Simple keyword matching fallback logic
                return self._fallback_answer(question)
                
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            formatted_error = f"""
            <div style="{StyleConstants.ERROR_STYLE}">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">⚠️</span>
                    <strong>Processing Error</strong>
                </div>
                <div style="font-size: 0.9rem; line-height: 1.5;">
                    {error_msg}
                </div>
            </div>
            """
            return {
                'summary_answer': error_msg,
                'detailed_answer': formatted_error
            }
    
    def _format_detailed_answer(self, markdown_answer: str, confidence: float, sources: list) -> str:
        """Format detailed answer with styling and metadata"""
        return f"""
        <div class="answer-container" style="
            background: #ffffff; 
            border-radius: 8px; 
            padding: 0; 
            margin-bottom: 15px; 
            border-left: 4px solid #007bff; 
            overflow-y: auto; 
            max-height: calc(100vh - 200px); 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        ">
            <div style="display: flex; align-items: center; padding: 20px 24px 15px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                <span style="font-size: 1.1rem; margin-right: 8px;">📋</span>
                <strong style="font-size: 1.05rem; color: #2c3e50;">Detailed Analysis Report</strong>
            </div>
            <div class="markdown-content document-style english-optimized">
                {markdown_answer}
            </div>
        </div>

        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; padding: 0 4px;">
            <div style="background: white; border-radius: 6px; padding: 8px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); flex: 1; min-width: 120px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="margin-right: 4px; font-size: 0.9rem;">📊</span>
                    <strong style="color: #2c3e50; font-size: 0.85rem;">Confidence</strong>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="background: {'#28a745' if confidence > 0.7 else '#ffc107' if confidence > 0.4 else '#dc3545'}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">
                        {confidence:.2f}
                    </span>
                    <span style="margin-left: 6px; color: #6c757d; font-size: 0.75rem;">
                        ({'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'})
                    </span>
                </div>
            </div>
            
            {f'''
            <div style="background: white; border-radius: 6px; padding: 8px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); flex: 1; min-width: 120px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="margin-right: 4px; font-size: 0.9rem;">📚</span>
                    <strong style="color: #2c3e50; font-size: 0.85rem;">Sources</strong>
                </div>
                <div style="color: #495057; font-size: 0.8rem;">
                    <span style="background: #e9ecef; padding: 1px 6px; border-radius: 8px; font-weight: 500;">
                        {len(sources)} documents
                    </span>
                </div>
            </div>
            ''' if sources else ''}
        </div>
        """
    
    def _get_fallback_answer(self, question: str, language: str) -> str:
        """Provide ADaM-related basic answers when RAG system is unavailable"""
        
        # ADaM-related knowledge base
        # 扩展的ADaM知识库（中英文版本）
        adam_knowledge_cn = {
            "adsl数据集": "ADSL（Subject-Level Analysis Dataset）是ADaM标准中的核心数据集，包含每个受试者的基线特征、人口统计学信息、处理分组和关键的汇总变量。必须包含USUBJID、STUDYID、SUBJID等关键变量。",
            "bds数据结构": "BDS（Basic Data Structure）是ADaM中用于纵向分析数据的标准结构，包含PARAMCD、PARAM、AVAL、BASE、CHG等核心分析变量，支持重复测量和时间序列分析。",
            "关键变量": "ADaM关键变量包括：USUBJID（唯一受试者标识）、PARAMCD（参数代码）、AVAL（分析值）、BASE（基线值）、CHG（较基线变化）、ABLFL（基线标志）、ANL01FL（分析标志）等。",
            "数据溯源性": "ADaM数据集必须建立完整的溯源性，通过--ORIG变量记录数据来源，确保每个分析变量都能追溯到原始SDTM数据集，满足监管审查要求。"
        }
        
        adam_knowledge_en = {
            "data structure": "ADaM (Analysis Data Model) is a CDISC standard data model for clinical trial statistical analysis, providing standardized data structures to support regulatory submissions.",
            "analysis variables": "ADaM datasets contain various types of analysis variables, such as PARAMCD (Parameter Code), AVAL (Analysis Value), BASE (Baseline Value), CHG (Change from Baseline), etc.",
            "baseline definition": "In ADaM datasets, baseline is typically identified through the ABLFL (Baseline Flag) variable, indicating the baseline visit used for analysis.",
            "traceability": "ADaM datasets must establish complete traceability through --ORIG variables to record data sources, ensuring each analysis variable can be traced back to original SDTM datasets."
        }
        
        # 根据语言选择知识库
        if language == "chinese":
            adam_knowledge = adam_knowledge_cn
        else:
            adam_knowledge = adam_knowledge_en
        
        # Match relevant knowledge based on question keywords
        question_lower = question.lower()
        for key, knowledge in adam_knowledge.items():
            if any(keyword in question_lower for keyword in key.split()):
                if language == "chinese":
                    return f"""<div class="fallback-answer">
                        <h4>💡 ADaM基础知识</h4>
                        <p>{knowledge}</p>
                        <p><em>注：这是基于ADaM标准的基础回答。如需更详细的信息，请确保RAG系统正常运行。</em></p>
                    </div>"""
                else:
                    return f"""<div class="fallback-answer">
                        <h4>💡 ADaM Basic Knowledge</h4>
                        <p>{knowledge}</p>
                        <p><em>Note: This is a basic answer based on ADaM standards. For more detailed information, please ensure the RAG system is running properly.</em></p>
                    </div>"""
        
        # Default answer
        if language == "chinese":
            return """<div class="fallback-answer">
                <h4>🤖 系统提示</h4>
                <p>抱歉，当前RAG系统不可用，无法提供详细的ADaM数据分析。</p>
                <p>请检查系统状态或稍后重试。</p>
            </div>"""
        else:
            return """<div class="fallback-answer">
                <h4>🤖 System Notice</h4>
                <p>Sorry, the RAG system is currently unavailable and cannot provide detailed ADaM data analysis.</p>
                <p>Please check the system status or try again later.</p>
            </div>"""
    
    def _fallback_answer(self, question: str) -> dict:
        """Fallback mode answer processing"""
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['hemoglobin', 'hemo', 'hgb']):
            answer = "**Hemoglobin (HEMO)** is a key variable for measuring blood oxygen-carrying capacity.\n\n- Usually measured in `g/dL` units\n- Important indicator for assessing anemia status"
        elif any(keyword in question_lower for keyword in ['mesa']):
            answer = "**MESA (Multi-Ethnic Study of Atherosclerosis)** is a longitudinal study examining cardiovascular disease development.\n\nKey features:\n- Multi-ethnic participation\n- Long-term follow-up\n- Cardiovascular disease prevention"
        else:
            answer = f"I found information related to **{question}**.\n\nPlease provide more specific questions for detailed answers."
        
        # Convert to HTML
        markdown_answer = markdown.markdown(answer, extensions=['extra', 'codehilite'])
        
        # Provide better formatting for fallback mode as well
        formatted_fallback = f"""
        <div class="answer-container" style="
            background: #ffffff; 
            border-radius: 8px; 
            padding: 0; 
            margin-bottom: 15px; 
            border-left: 4px solid #6c757d; 
            overflow-y: auto; 
            max-height: calc(100vh - 200px); 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        ">
            <div style="display: flex; align-items: center; padding: 20px 24px 15px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                <span style="font-size: 1.2rem; margin-right: 8px;">💭</span>
                <strong style="font-size: 1.1rem; color: #2c3e50;">Basic Answer</strong>
            </div>
            <div class="markdown-content document-style english-optimized">
                {markdown_answer}
            </div>
        </div>

        <div style="{StyleConstants.WARNING_STYLE} text-align: center; font-size: 0.85rem; margin-top: 15px;">
            ⚠️ Currently in demo mode, recommend enabling RAG system for more accurate answers
        </div>
        """
        
        return {
            'summary_answer': answer,
            'detailed_answer': formatted_fallback
        }

    def _generate_fallback_summary(self, detailed_answer: str) -> str:
        """Generate a fallback summary when RAG system summary fails"""
        if not detailed_answer or detailed_answer == 'No answer generated':
            return "Unable to generate summary"
        
        # 简单的摘要生成：取前两句话
        import re
        # 移除HTML标签
        clean_text = re.sub(r'<[^>]+>', '', detailed_answer)
        # 按句号分割，取前两句
        sentences = re.split(r'[.!?]+', clean_text)
        summary_sentences = [s.strip() for s in sentences[:2] if s.strip()]
        
        if summary_sentences:
            return '. '.join(summary_sentences) + '.'
        else:
            return "Summary generation failed, please check detailed answer."