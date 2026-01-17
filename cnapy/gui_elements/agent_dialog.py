"""AI Agent Dialog for CNApy

This module provides the chat interface for the CNApy Multi-Agent System.
Users can interact with the system using natural language commands in
Korean or English.

Features:
- Chat-based interface
- Quick action buttons for common operations
- Agent filter tabs
- Real-time execution feedback
- LLM configuration integration
"""

import traceback

from qtpy.QtCore import Qt, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cnapy.agents.base_agent import AgentContext, AgentResponse
from cnapy.agents.orchestrator_agent import OrchestratorAgent
from cnapy.appdata import AppData
from cnapy.gui_elements.llm_analysis_dialog import LLMConfig


class AgentWorkerThread(QThread):
    """Worker thread for executing agent requests without blocking the UI."""

    result_ready = Signal(object)  # AgentResponse
    error_occurred = Signal(str)
    status_update = Signal(str)

    def __init__(self, orchestrator: OrchestratorAgent, message: str):
        super().__init__()
        self.orchestrator = orchestrator
        self.message = message

    def run(self):
        """Execute the agent request."""
        try:
            self.status_update.emit("Processing...")
            response = self.orchestrator.route(self.message)
            self.result_ready.emit(response)
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class ChatMessage(QFrame):
    """Widget for displaying a single chat message."""

    def __init__(self, role: str, content: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.setup_ui(content)

    def setup_ui(self, content: str):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Role indicator
        if self.role == "user":
            role_text = "You"
            self.setStyleSheet(
                """
                ChatMessage {
                    background-color: #e3f2fd;
                    border-radius: 10px;
                    margin-left: 50px;
                }
            """
            )
        elif self.role == "assistant":
            role_text = "AI"
            self.setStyleSheet(
                """
                ChatMessage {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin-right: 50px;
                }
            """
            )
        else:  # system
            role_text = "System"
            self.setStyleSheet(
                """
                ChatMessage {
                    background-color: #fff3e0;
                    border-radius: 10px;
                }
            """
            )

        content_layout = QVBoxLayout()

        # Role label
        role_label = QLabel(f"<b>{role_text}</b>")
        content_layout.addWidget(role_label)

        # Message content
        message_label = QLabel(content)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(message_label)

        layout.addLayout(content_layout)
        self.setLayout(layout)


class AgentDialog(QDialog):
    """Main dialog for AI Agent interaction."""

    def __init__(self, appdata: AppData, main_window=None):
        super().__init__()
        self.appdata = appdata
        self.main_window = main_window
        self.llm_config = LLMConfig()
        self.worker_thread: AgentWorkerThread | None = None

        # Initialize agent context and orchestrator
        self.context = AgentContext(
            appdata=appdata,
            main_window=main_window,
        )
        self.orchestrator = OrchestratorAgent(self.context, self.llm_config)

        self.setWindowTitle("CNApy AI Agent")
        self.setMinimumSize(800, 600)
        self.setup_ui()

        # Add welcome message
        self._add_assistant_message(
            "Hello! I'm the CNApy AI Agent. I can help you with metabolic model analysis.\n\n"
            "안녕하세요! CNApy AI 에이전트입니다. 대사 모델 분석을 도와드릴 수 있습니다.\n\n"
            "Try commands like:\n"
            "- 'Perform FBA' / 'FBA 수행해줘'\n"
            "- 'Set anaerobic condition' / '혐기 조건 설정해줘'\n"
            "- 'Find essential genes' / '필수 유전자 찾아줘'"
        )

    def setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout()

        # Header with title and config button
        header_layout = QHBoxLayout()
        title_label = QLabel("<h2>🤖 CNApy AI Agent</h2>")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        config_btn = QToolButton()
        config_btn.setText("⚙️ Config")
        config_btn.setToolTip("Configure LLM API settings")
        config_btn.clicked.connect(self._show_config)
        header_layout.addWidget(config_btn)
        main_layout.addLayout(header_layout)

        # Agent filter tabs
        self.agent_tabs = QTabWidget()
        self.agent_tabs.addTab(QWidget(), "All")
        self.agent_tabs.addTab(QWidget(), "Flux")
        self.agent_tabs.addTab(QWidget(), "Gene")
        self.agent_tabs.addTab(QWidget(), "Scenario")
        self.agent_tabs.addTab(QWidget(), "Query")
        self.agent_tabs.addTab(QWidget(), "Strain")
        self.agent_tabs.setTabPosition(QTabWidget.North)
        self.agent_tabs.setMaximumHeight(30)
        main_layout.addWidget(self.agent_tabs)

        # Splitter for chat and quick actions
        splitter = QSplitter(Qt.Vertical)

        # Chat area
        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_content)

        chat_layout.addWidget(self.chat_scroll)
        chat_widget.setLayout(chat_layout)
        splitter.addWidget(chat_widget)

        # Quick actions panel
        quick_actions_widget = QWidget()
        quick_layout = QVBoxLayout()
        quick_layout.setContentsMargins(5, 5, 5, 5)

        quick_label = QLabel("<b>📌 Quick Actions</b>")
        quick_layout.addWidget(quick_label)

        # Quick action buttons
        actions_container = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)

        quick_actions = [
            ("FBA", "FBA 수행", "perform_fba"),
            ("pFBA", "pFBA 수행", "perform_pfba"),
            ("FVA", "FVA 분석", "perform_fva"),
            ("Aerobic", "호기 조건", "aerobic"),
            ("Anaerobic", "혐기 조건", "anaerobic"),
            ("Essential", "필수 유전자", "essential_genes"),
            ("Clear", "초기화", "clear_scenario"),
        ]

        for en_label, ko_label, action_id in quick_actions:
            btn = QPushButton(f"{en_label}\n{ko_label}")
            btn.setMaximumWidth(100)
            btn.setMinimumHeight(50)
            btn.clicked.connect(lambda checked, a=action_id: self._execute_quick_action(a))
            actions_layout.addWidget(btn)

        actions_container.setLayout(actions_layout)
        quick_layout.addWidget(actions_container)

        quick_actions_widget.setLayout(quick_layout)
        quick_actions_widget.setMaximumHeight(100)
        splitter.addWidget(quick_actions_widget)

        splitter.setSizes([500, 100])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(5)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your command... / 명령을 입력하세요...")
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send 📤")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumWidth(80)
        input_layout.addWidget(self.send_btn)

        main_layout.addLayout(input_layout)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def _add_message(self, role: str, content: str):
        """Add a message to the chat."""
        # Insert before the stretch
        message_widget = ChatMessage(role, content)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_widget)

        # Scroll to bottom
        QApplication.processEvents()
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _add_user_message(self, content: str):
        """Add a user message to the chat."""
        self._add_message("user", content)

    def _add_assistant_message(self, content: str):
        """Add an assistant message to the chat."""
        self._add_message("assistant", content)

    def _add_system_message(self, content: str):
        """Add a system message to the chat."""
        self._add_message("system", content)

    @Slot()
    def _send_message(self):
        """Send the user's message to the orchestrator."""
        message = self.input_field.text().strip()
        if not message:
            return

        # Clear input and disable while processing
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)

        # Add user message to chat
        self._add_user_message(message)

        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setText("🔄 Processing...")

        # Execute in background thread
        self.worker_thread = AgentWorkerThread(self.orchestrator, message)
        self.worker_thread.result_ready.connect(self._on_result)
        self.worker_thread.error_occurred.connect(self._on_error)
        self.worker_thread.status_update.connect(self._on_status_update)
        self.worker_thread.finished.connect(self._on_finished)
        self.worker_thread.start()

    @Slot(object)
    def _on_result(self, response: AgentResponse):
        """Handle the agent response."""
        # Format the response message
        message = response.get_message(self.context.current_language)

        # Add status indicator
        if response.success:
            status = "✅"
        else:
            status = "❌"

        # Add agent info if available
        if response.agent_name:
            agent_info = f"[{response.agent_name}] "
        else:
            agent_info = ""

        full_message = f"{status} {agent_info}{message}"
        self._add_assistant_message(full_message)

        # Update the main window if needed
        if response.success and self.main_window:
            try:
                self.main_window.centralWidget().update()
            except Exception:
                pass

    @Slot(str)
    def _on_error(self, error: str):
        """Handle errors from the agent."""
        self._add_system_message(f"❌ Error: {error}")

    @Slot(str)
    def _on_status_update(self, status: str):
        """Handle status updates."""
        self.status_label.setText(f"🔄 {status}")

    @Slot()
    def _on_finished(self):
        """Handle completion of the worker thread."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("")
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()

    def _execute_quick_action(self, action_id: str):
        """Execute a quick action."""
        # Map action IDs to natural language commands
        action_commands = {
            "perform_fba": "Perform FBA",
            "perform_pfba": "Perform pFBA",
            "perform_fva": "Perform FVA with 90% optimum",
            "aerobic": "Set aerobic condition",
            "anaerobic": "Set anaerobic condition",
            "essential_genes": "Find essential genes",
            "clear_scenario": "Clear scenario",
        }

        command = action_commands.get(action_id, action_id)
        self.input_field.setText(command)
        self._send_message()

    @Slot()
    def _show_config(self):
        """Show the LLM configuration dialog."""
        from cnapy.gui_elements.llm_analysis_dialog import LLMAnalysisDialog

        dialog = LLMAnalysisDialog(self.appdata)
        dialog.tabs.setCurrentIndex(0)  # Show config tab
        dialog.exec()
        # Reload config
        self.llm_config = LLMConfig()
        self.orchestrator.llm_config = self.llm_config

    def closeEvent(self, event):
        """Handle dialog close."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.wait(1000)
        event.accept()
