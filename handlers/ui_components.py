from shiny import ui
from .utils import StyleConstants

class UIComponents:
    """UI component rendering class for creating interface elements"""
    
    @staticmethod
    def chat_system_status(system_ready: bool):
        """Display chat system status"""
        status_style = "padding: 10px 15px; border-radius: 8px; font-size: 0.9rem; margin-bottom: 15px; font-weight: 600;"
        if system_ready:
            return ui.div(
                "✅ RAG System Ready",
                style=f"{status_style} {StyleConstants.SUCCESS_STYLE}"
            )
        else:
            return ui.div(
                "⚠️ Demo Mode",
                style=f"{status_style} {StyleConstants.WARNING_STYLE}"
            )
    
    @staticmethod
    def chat_history(messages: list):
        """Display chat history"""
        if not messages:
            return UIComponents._render_welcome_screen()
        
        return ui.div(*[UIComponents._render_message(msg) for msg in messages])
    
    @staticmethod
    def _render_welcome_screen():
        """Render welcome screen with example questions"""
        example_questions = [
            ("example1", "📊 What are the key variables and their definitions in ADSL (Subject-Level Analysis Dataset)?", "#e3f2fd", "#1565c0", "#90caf9"),
            ("example2", "🔍 What are the requirements for PARAMCD and PARAM variables in BDS datasets?", "#e8f5e8", "#2e7d32", "#81c784"),
            ("example3", "📋 What are the main differences between ADSL and BDS (Basic Data Structure) datasets?", "#fff3e0", "#ef6c00", "#ffcc02")
        ]
        
        return ui.div(
            ui.div(
                "👋 Welcome! Please ask any questions about ADaM data standards, variables, or analysis datasets.",
                style="color: #6c757d; text-align: center; padding: 20px; font-style: italic;"
            ),
            ui.div(
                "💡 Quick Start - Click questions below:",
                style="color: #495057; text-align: center; font-weight: 600; margin-bottom: 15px; font-size: 0.9rem;"
            ),
            ui.div(
                {"style": "display: flex; flex-direction: column; gap: 10px; padding: 0 20px 20px;"},
                *[ui.input_action_button(
                    btn_id, text,
                    style=f"width: 100%; padding: 12px 16px; background: {bg}; color: {color}; border: 1px solid {border}; border-radius: 8px; font-size: 0.9rem; text-align: left; transition: all 0.2s ease;"
                ) for btn_id, text, bg, color, border in example_questions]
            )
        )
    
    @staticmethod
    def _render_message(msg):
        """Render individual message"""
        if msg['type'] == 'user':
            return ui.div(
                {"style": "margin: 12px 0; text-align: right;"},
                ui.div(
                    msg['content'],
                    style="display: inline-block; background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); color: white; padding: 12px 18px; border-radius: 18px 18px 6px 18px; max-width: 80%; word-wrap: break-word; box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3); font-weight: 500;"
                )
            )
        else:
            # Assistant消息使用纯文本格式，保持换行
            return ui.div(
                {"style": "margin: 12px 0; text-align: left;"},
                ui.div(
                    msg['content'],  # 直接使用纯文本内容
                    style="display: inline-block; background: #ffffff; padding: 12px 16px; border-radius: 12px; max-width: 85%; word-wrap: break-word; box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15); border: 1px solid rgba(33, 150, 243, 0.1); line-height: 1.6; color: #2c3e50; white-space: pre-wrap;"
                )
            )
    
    @staticmethod
    def current_answer_details(answer: str):
        """Display current answer details"""
        if answer:
            return ui.div(
                ui.HTML(answer),
                style="padding: 15px; line-height: 1.6;"
            )
        else:
            return ui.div(
                "💡 Ask a question to view detailed analysis here.",
                style="color: #6c757d; padding: 15px; text-align: center; font-style: italic;"
            )
    
    @staticmethod
    def history_indicator(history: list, index: int):
        """Display current viewing history record indicator"""
        if not history:
            return ui.div()
        
        if index == -1:
            return ui.div(
                "📍 Latest Answer",
                style="color: #28a745; font-size: 0.85rem; font-weight: 500;"
            )
        else:
            return ui.div(
                f"📚 History {index + 1}/{len(history)}",
                style="color: #6c757d; font-size: 0.85rem; font-weight: 500;"
            )
    
    @staticmethod
    def history_navigation(history: list, index: int):
        """History navigation controls"""
        if len(history) <= 1:
            return ui.div(
                "💭 No History Available",
                style="text-align: center; color: #6c757d; font-size: 0.85rem; padding: 5px;"
            )
        
        total = len(history)
        # 修正页码逻辑：index=-1时显示第n页(Latest)，index=0时显示第1页，以此类推
        current_page_display = "Latest" if index == -1 else f"{index + 1} / {total}"
        
        return ui.div(
            {"style": "display: flex; justify-content: space-between; align-items: center;"},
            
            # Left: Previous button
            ui.input_action_button(
                "prev_answer",
                "◀ Previous",
                style="padding: 6px 12px; font-size: 0.85rem; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;"
            ),
            
            # Center: Page indicator, back to latest button, and page search input
            ui.div(
                {"style": "display: flex; gap: 10px; align-items: center;"},
                ui.div(
                    current_page_display,
                    style="color: #495057; font-size: 0.85rem; font-weight: 500;"
                ),
                ui.input_action_button(
                    "goto_latest",
                    "📍 Latest",
                    style="padding: 4px 8px; font-size: 0.8rem; background: #28a745; color: white; border: none; border-radius: 4px;"
                ) if index != -1 else ui.div(),
                # 页面搜索栏放在这里
                ui.div(
                    {"style": "display: flex; align-items: center; gap: 4px;"},
                    ui.div(
                        "🔍",
                        style="color: #495057; font-size: 0.8rem;"
                    ),
                    ui.div(
                        {"style": "display: inline-block;"},
                        ui.div(
                            {"style": "height:28px; display: flex; align-items: center;"},
                            ui.input_text(
                                "page_jump_input",
                                "",
                                placeholder=f"1-{total}",
                                width="50px"
                            )
                        )
                    )
                )
            ),
            
            # Right: Next button (恢复原位置)
            ui.input_action_button(
                "next_answer",
                "Next ▶",
                style="padding: 6px 12px; font-size: 0.85rem; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;"
            )
        )
    
    @staticmethod
    def welcome_screen():
        # Example questions for ADaM data
        examples = [
            ("example1", "📊 What are the key variables and their definitions in ADSL (Subject-Level Analysis Dataset)?", "#e3f2fd", "#1565c0", "#90caf9"),
            ("example2", "🔍 What are the requirements for PARAMCD and PARAM variables in BDS datasets?", "#e8f5e8", "#2e7d32", "#81c784"),
            ("example3", "📋 What are the main differences between ADSL and BDS (Basic Data Structure) datasets?", "#fff3e0", "#ef6c00", "#ffcc02")
        ]
        
        return ui.div(
            ui.div(
                "👋 Welcome! Please ask any questions about ADaM data standards, variables, or analysis datasets.",
                style="color: #6c757d; text-align: center; padding: 20px; font-style: italic;"
            ),
            ui.div(
                "💡 Quick Start - Click questions below:",
                style="color: #495057; text-align: center; font-weight: 600; margin-bottom: 15px; font-size: 0.9rem;"
            ),
            ui.div(
                {"style": "display: flex; flex-direction: column; gap: 10px; padding: 0 20px 20px;"},
                *[ui.input_action_button(
                    btn_id, text,
                    style=f"width: 100%; padding: 12px 16px; background: {bg}; color: {color}; border: 1px solid {border}; border-radius: 8px; font-size: 0.9rem; text-align: left; transition: all 0.2s ease;"
                ) for btn_id, text, bg, color, border in example_questions]
            )
        )