from shiny import reactive

class HistoryHandlers:
    """History record processing class for managing answer history"""
    
    def __init__(self):
        self.answer_history = reactive.Value([])
        self.current_history_index = reactive.Value(-1)
    
    def add_to_history(self, question: str, detailed_answer: str, summary_answer: str, timestamp):
        """Add new Q&A record to history"""
        history = self.answer_history.get().copy()
        history.append({
            'question': question,
            'detailed_answer': detailed_answer,
            'summary_answer': summary_answer,
            'timestamp': timestamp,
            'index': len(history)
        })
        self.answer_history.set(history)
        self.current_history_index.set(-1)  # Reset to latest record
    
    def clear_history(self):
        """Clear history records"""
        self.answer_history.set([])
        self.current_history_index.set(-1)
    
    def get_current_display_answer(self, current_answer):
        """Get the answer details that should currently be displayed"""
        history = self.answer_history.get()
        index = self.current_history_index.get()
        
        if not history:
            return current_answer.get()
        
        if index == -1:  # Display latest
            return current_answer.get()
        elif 0 <= index < len(history):
            return history[index]['detailed_answer']
        else:
            return current_answer.get()
    
    def setup_navigation_handlers(self, input):
        """Setup history record navigation events"""
        
        @reactive.Effect
        @reactive.event(input.prev_answer)
        def handle_prev_answer():
            """Switch to previous answer"""
            history = self.answer_history.get()
            current_index = self.current_history_index.get()
            
            if not history:
                return
            
            if current_index == -1:  # Currently at latest, switch to second-to-last history record
                if len(history) >= 2:
                    new_index = len(history) - 2
                else:
                    new_index = 0
            elif current_index > 0:
                new_index = current_index - 1
            else:
                new_index = current_index  # Already at first, no change
            
            self.current_history_index.set(new_index)
        
        @reactive.Effect
        @reactive.event(input.next_answer)
        def handle_next_answer():
            """Switch to next answer"""
            history = self.answer_history.get()
            current_index = self.current_history_index.get()
            
            if not history:
                return
            
            if current_index == -1:  # Currently at latest, cannot go to next
                return
            elif current_index < len(history) - 2:  # Not second-to-last
                new_index = current_index + 1
            else:  # Is second-to-last, jump to latest answer
                new_index = -1
            
            self.current_history_index.set(new_index)
        
        @reactive.Effect
        @reactive.event(input.goto_latest)
        def handle_goto_latest():
            """Return to latest answer"""
            self.current_history_index.set(-1)
    
    def jump_to_page(self, page_num: int):
        """Jump to specific page number"""
        history = self.answer_history.get()
        
        if not history:
            return
        
        try:
            # 修正页码逻辑：
            # 页码1到n-1对应历史记录index 0到n-2
            # 页码n对应latest (index=-1)
            total_pages = len(history)
            
            if 1 <= page_num < total_pages:
                # 跳转到历史记录（页码1对应index 0）
                target_index = page_num - 1
                self.current_history_index.set(target_index)
            elif page_num == total_pages:
                # 跳转到最新（latest，第n页）
                self.current_history_index.set(-1)
        except (ValueError, TypeError):
            # 输入不是有效数字，忽略
            pass