from shiny import App, ui, render, reactive, Inputs, Outputs, Session
import asyncio
from datetime import datetime
from pathlib import Path

# Import project modules
from config.settings import ADaMConfig
from rag.system import ADaMRAGSystem
from handlers import UIComponents, MessageHandlers, HistoryHandlers, QuestionProcessor
from handlers.utils import UIUtils

# Simplified system initialization
config = ADaMConfig()
rag_system = None
system_ready = False

try:
    rag_system = ADaMRAGSystem()
    system_ready = True
    print("✅ ADaM RAG system initialized successfully")
except Exception as e:
    print(f"⚠️ ADaM RAG system initialization failed: {e}")
    print("🔄 Using demo mode")

app_ui = UIUtils.create_app_ui()

def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values
    chat_messages = reactive.Value([])
    current_answer = reactive.Value("")
    is_processing = reactive.Value(False)
    output_language = reactive.Value("english")
    
    # Initialize processors
    question_processor = QuestionProcessor(rag_system, system_ready)
    history_manager = HistoryHandlers()
    message_handlers = MessageHandlers(question_processor, history_manager)
    
    # Language toggle handler
    @reactive.Effect
    @reactive.event(input.toggle_language)
    def handle_language_toggle():
        current_lang = output_language.get()
        new_lang = "chinese" if current_lang == "english" else "english"
        output_language.set(new_lang)
    
    # Setup handlers
    message_handlers.setup_handlers(input, chat_messages, current_answer, is_processing, output_language)
    history_manager.setup_navigation_handlers(input)
    

    @reactive.Effect
    @reactive.event(input.page_jump_input)
    def handle_page_jump():
        """Handle page jump when user presses Enter"""
        page_input = input.page_jump_input()
        if page_input and page_input.startswith('ENTER:'):
            try:
                page_num_str = page_input.replace('ENTER:', '').strip()
                page_num = int(page_num_str)
                history_manager.jump_to_page(page_num)
            except ValueError:
                pass  
    
    # UI rendering functions
    @output
    @render.ui
    def chat_system_status():
        return UIComponents.chat_system_status(system_ready)
    
    @output
    @render.ui
    def chat_history():
        return UIComponents.chat_history(chat_messages.get())
    
    @output
    @render.ui
    def current_answer_details():
        answer = history_manager.get_current_display_answer(current_answer)
        return UIComponents.current_answer_details(answer)
    
    @output
    @render.ui
    def history_indicator():
        return UIComponents.history_indicator(
            history_manager.answer_history.get(), 
            history_manager.current_history_index.get()
        )
    
    @output
    @render.ui
    def history_navigation():
        return UIComponents.history_navigation(
            history_manager.answer_history.get(), 
            history_manager.current_history_index.get()
        )

app = App(app_ui, server, static_assets=Path(__file__).parent / "static")

if __name__ == "__main__":
    app.run()


