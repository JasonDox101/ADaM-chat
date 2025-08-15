from shiny import ui

# 添加统一样式常量类
class StyleConstants:
    """统一的样式常量定义"""
    
    # 统一的强调样式
    HIGHLIGHT_STYLE = "background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%); color: #92400e; padding: 0.1em 0.3em; border-radius: 3px; font-weight: 600; border: 1px solid #f59e0b;"
    
    # 统一的警告样式
    WARNING_STYLE = "background: #fff3cd; color: #856404; border: 1px solid #fbbf24; padding: 12px 16px; border-radius: 8px;"
    
    # 统一的错误样式
    ERROR_STYLE = "background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 8px;"
    
    # 统一的成功样式
    SUCCESS_STYLE = "background: #c8e6c9; color: #2e7d32; border: 1px solid #81c784;"
    
    # 统一的信息样式
    INFO_STYLE = "background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;"

class UIUtils:
    """UI utility class for creating application interface components"""
    
    @staticmethod
    def create_app_ui():
        """Create application UI - moved from original app_ui"""
        return ui.page_fluid(
            # Include external CSS file
            ui.tags.link(rel="stylesheet", href="styles.css"),
            
            # Top header area
            ui.div(
                {"class": "top-header"},
                ui.div(
                    {"style": "display: flex; align-items: center; justify-content: space-between;"},
                    # Left: Sidebar toggle button
                    ui.div(
                        ui.input_action_button(
                            "toggle_sidebar",
                            "",
                            class_="sidebar-toggle",
                            style="background: none; border: none; font-size: 1.5rem; color: #451a03; cursor: pointer; padding: 8px; border-radius: 4px; transition: all 0.3s ease;"
                        ),
                        style="flex: 0 0 auto;"
                    ),
                    # Center: Title content
                    ui.div(
                        [
                            ui.h2("🧠 ADaM Data Q&A Assistant"),
                            ui.p("Ask questions about ADaM variables, datasets, or analysis standards")
                        ],
                        style="flex: 1; text-align: center;"
                    ),
                    # Right: Language toggle button - 简化版本
                    ui.div(
                        ui.div(
                            {
                                "class": "language-toggle-container",
                                "id": "language_toggle_container",
                                "style": "position: relative; width: 60px; height: 30px; background: #f8f9fa; border-radius: 15px; cursor: pointer; transition: all 0.3s ease; border: 2px solid #451a03; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);"
                            },
                            [
                                # 滑动指示器 - 只保留这个
                                ui.div(
                                    {
                                        "class": "language-slider",
                                        "id": "language_slider",
                                        "style": "position: absolute; top: 2px; left: 2px; width: 24px; height: 24px; background: linear-gradient(135deg, #451a03 0%, #6d2710 100%); border-radius: 12px; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold; box-shadow: 0 2px 4px rgba(69, 26, 3, 0.3);"
                                    },
                                    "EN"
                                ),
                                # 隐藏的按钮用于事件处理
                                ui.input_action_button(
                                    "toggle_language",
                                    "",
                                    style="position: absolute; width: 100%; height: 100%; opacity: 0; cursor: pointer; border: none; background: transparent;"
                                )
                            ]
                        ),
                        style="flex: 0 0 auto;"
                    )
                )
            ),
            
            # Main content area
            ui.div(
                {"class": "main-content", "id": "main-content", "style": "display: flex; height: calc(100vh - 80px); transition: all 0.3s ease;"},
                
                # Left sidebar (original chat panel)
                ui.div(
                    {
                        "class": "sidebar", 
                        "id": "sidebar",
                        "style": "width: 400px; min-width: 400px; display: flex; flex-direction: column; background: #fed7aa; border-right: 1px solid #ea580c; padding: 15px; transition: all 0.3s ease; overflow: hidden;"
                    },
                    
                    # System status
                    ui.output_ui("chat_system_status"),
                    
                    # Chat history area
                    ui.div(
                        {"class": "chat-history"},
                        ui.output_ui("chat_history")
                    ),
                    
                    # Input area
                    ui.div(
                        {"class": "input-area"},
                        ui.div(
                            {"style": "position: relative;"},
                            ui.input_text_area(
                                "chat_input",
                                "",
                                placeholder="💬 Enter your questions about ADaM data... (Press Enter to send, Shift+Enter for new line)",
                                height="100px",
                                width="100%"
                            )
                        ),
                        ui.div(
                            {"style": "margin-top: 18px; display: flex; justify-content: space-between; align-items: center;"},
                            ui.div(
                                # Empty placeholder to keep buttons right-aligned
                                style="flex: 1;"
                            ),
                            ui.div(
                                [
                                    ui.input_action_button(
                                        "send_message",
                                        "📤 Send",
                                        class_="btn-primary"
                                    ),
                                    ui.input_action_button(
                                        "clear_chat",
                                        "🗑️ Clear",
                                        class_="btn-secondary"
                                    )
                                ]
                            )
                        ),
                        # Tip text moved below buttons
                        ui.div(
                            "💡 Tip: Press Enter to send, Shift+Enter for new line",
                            style="color: #78909c; font-size: 12px; font-weight: 500; text-align: center; margin-top: 10px;"
                        )
                    )
                ),
                
                # Right main panel
                ui.div(
                    {
                        "id": "main-panel",
                        "style": "flex: 1; display: flex; flex-direction: column; height: 100%; min-height: calc(100vh - 80px); padding: 0; transition: all 0.3s ease; background: #fffbf5;"
                    },
                    
                    # Current answer details
                    ui.card(
                        ui.card_header(
                            ui.div(
                                {"style": "display: flex; justify-content: space-between; align-items: center;"},
                                ui.div("🎯 Answer Details"),
                                ui.output_ui("history_indicator")  # Show which answer is currently being viewed
                            )
                        ),
                        ui.div(
                            {"style": "flex: 1; overflow-y: auto; padding: 5px; height: 100%;"},
                            ui.output_ui("current_answer_details")
                        ),
                        # History navigation control panel
                        ui.div(
                            {"style": "padding: 10px 15px; border-top: 1px solid #e2e8f0; background: #f8fafc;"},
                            ui.output_ui("history_navigation")
                        )
                    )
                )
            ),
            
            # CSS styles
            ui.tags.style("""
                /* ADaM Data Q&A Assistant - Sidebar toggle button styles - improved icon */
                .sidebar-toggle:before {
                    content: '◧';
                    font-weight: bold;
                }
                
                .sidebar-toggle:hover {
                    background: rgba(69, 26, 3, 0.1) !important;
                    transform: scale(1.05);
                }
                
                /* Language toggle container styles - 简化版 */
                .language-toggle-container {
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
                }
                
                .language-toggle-container:hover {
                    box-shadow: 0 3px 8px rgba(69, 26, 3, 0.2), inset 0 1px 3px rgba(0,0,0,0.1) !important;
                    transform: translateY(-1px);
                }
                
                /* 中文模式样式 */
                .language-toggle-container.chinese .language-slider {
                    left: 32px !important;
                    background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
                }
                
                /* 滑块动画效果 */
                .language-slider {
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                }
                
                /* 响应式优化 */
                @media (max-width: 768px) {
                    .language-toggle-container {
                        width: 50px !important;
                        height: 26px !important;
                    }
                    
                    .language-slider {
                        width: 20px !important;
                        height: 20px !important;
                        font-size: 9px !important;
                    }
                    
                    .language-toggle-container.chinese .language-slider {
                        left: 28px !important;
                    }
                }
                
                .language-toggle-container.chinese .language-slider {
                    content: '中';
                }
                
                /* Sidebar collapsed state */
                .sidebar.collapsed {
                    width: 0 !important;
                    min-width: 0 !important;
                    padding: 0 !important;
                    border-right: none !important;
                    overflow: hidden !important;
                }
                
                /* Main panel expanded state */
                .main-panel.expanded {
                    width: 100% !important;
                }
                
                /* Responsive adjustments */
                @media (max-width: 768px) {
                    .sidebar {
                        position: absolute;
                        left: 0;
                        top: 0;
                        height: 100%;
                        z-index: 1000;
                        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                    }
                    
                    .sidebar.collapsed {
                        left: -400px;
                    }
                    
                    .language-toggle-container {
                        width: 60px !important;
                        height: 28px !important;
                    }
                    
                    .language-slider {
                        width: 28px !important;
                        height: 20px !important;
                    }
                }
            """),
            
            # JavaScript code
            ui.tags.script("""
                document.addEventListener('DOMContentLoaded', function() {
                    // Keyboard event listener for ADaM data chat input
                    const chatInput = document.getElementById('chat_input');
                    
                    if (chatInput) {
                        chatInput.addEventListener('keydown', function(event) {
                            if (event.key === 'Enter' && !event.shiftKey) {
                                event.preventDefault();
                                const sendButton = document.getElementById('send_message');
                                if (sendButton && chatInput.value.trim()) {
                                    sendButton.click();
                                }
                            }
                        });
                    }
                    
                    // 页面跳转输入框的Enter键支持 - 使用更可靠的方法
                    function setupPageJumpInput() {
                        const pageInput = document.getElementById('page_jump_input');
                        if (pageInput) {
                            // 移除之前的事件监听器
                            pageInput.removeEventListener('keydown', pageInput._pageJumpHandler);
                            
                            // 创建新的事件处理器
                            pageInput._pageJumpHandler = function(event) {
                                if (event.key === 'Enter' && !event.shiftKey) {
                                    event.preventDefault();
                                    const pageValue = pageInput.value.trim();
                                    if (pageValue && /^\d+$/.test(pageValue)) {
                                        // 使用Shiny.setInputValue直接设置值
                                        if (window.Shiny && window.Shiny.setInputValue) {
                                            window.Shiny.setInputValue('page_jump_input', 'ENTER:' + pageValue, {priority: 'event'});
                                            
                                            // 延迟清空输入框
                                            setTimeout(() => {
                                                pageInput.value = '';
                                            }, 100);
                                        } else {
                                            // 备用方法：直接设置值并触发change事件
                                            pageInput.value = 'ENTER:' + pageValue;
                                            const changeEvent = new Event('change', { bubbles: true });
                                            pageInput.dispatchEvent(changeEvent);
                                            
                                            setTimeout(() => {
                                                pageInput.value = '';
                                                const clearEvent = new Event('change', { bubbles: true });
                                                pageInput.dispatchEvent(clearEvent);
                                            }, 100);
                                        }
                                    }
                                }
                            };
                            
                            // 添加事件监听器
                            pageInput.addEventListener('keydown', pageInput._pageJumpHandler);
                        }
                    }
                    
                    // Sidebar toggle functionality
                    let sidebarCollapsed = false;
                    const toggleButton = document.getElementById('toggle_sidebar');
                    const sidebar = document.getElementById('sidebar');
                    const mainPanel = document.getElementById('main-panel');
                    
                    if (toggleButton && sidebar && mainPanel) {
                        toggleButton.addEventListener('click', function() {
                            sidebarCollapsed = !sidebarCollapsed;
                            
                            if (sidebarCollapsed) {
                                sidebar.classList.add('collapsed');
                                mainPanel.classList.add('expanded');
                                toggleButton.style.transform = 'rotate(180deg)';
                            } else {
                                sidebar.classList.remove('collapsed');
                                mainPanel.classList.remove('expanded');
                                toggleButton.style.transform = 'rotate(0deg)';
                            }
                        });
                    }
                    
                    // Keyboard shortcut support (Ctrl+B to toggle sidebar)
                    document.addEventListener('keydown', function(event) {
                        if (event.ctrlKey && event.key === 'b') {
                            event.preventDefault();
                            if (toggleButton) {
                                toggleButton.click();
                            }
                        }
                    });
                    
                    // ⚡ 简化JavaScript逻辑
                    const languageContainer = document.getElementById('language_toggle_container');
                    const languageSlider = document.getElementById('language_slider');
                    
                    if (languageContainer && languageSlider) {
                        let isEnglish = true;
                        
                        languageContainer.addEventListener('click', function() {
                            isEnglish = !isEnglish;
                            
                            if (isEnglish) {
                                languageContainer.classList.remove('chinese');
                                languageSlider.style.left = '2px';
                                languageSlider.textContent = 'EN';
                                languageSlider.style.background = 'linear-gradient(135deg, #451a03 0%, #6d2710 100%)';
                            } else {
                                languageContainer.classList.add('chinese');
                                languageSlider.style.left = '32px';
                                languageSlider.textContent = '中';
                                languageSlider.style.background = 'linear-gradient(135deg, #d97706 0%, #b45309 100%)';
                            }
                        });
                    }
                    
                    // 初始设置页面跳转输入框
                    setupPageJumpInput();
                    
                    // 监听DOM变化，确保动态生成的输入框也有事件监听
                    const observer = new MutationObserver(function(mutations) {
                        mutations.forEach(function(mutation) {
                            if (mutation.type === 'childList') {
                                setupPageJumpInput();
                            }
                        });
                    });
                    
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                    
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                });
            """),
            
            fillable=True
        )