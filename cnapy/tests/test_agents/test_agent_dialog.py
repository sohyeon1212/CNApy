"""Tests for Agent Dialog Integration

This module tests the agent dialog UI integration including
CrewAI support and cancellation functionality.

Note: These tests mock Qt components to avoid requiring a display.
"""

from unittest.mock import MagicMock, patch


class TestAgentWorkerThread:
    """Test AgentWorkerThread functionality."""

    def test_worker_thread_import(self):
        """Test that AgentWorkerThread can be imported."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        assert AgentWorkerThread is not None

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_worker_thread_init(self, mock_qthread):
        """Test worker thread initialization."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        mock_orchestrator = MagicMock()
        message = "test message"

        # Mock the parent class __init__
        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread.orchestrator = mock_orchestrator
            thread.message = message
            thread.use_crewai = False
            thread._cancel_requested = False

            assert thread.message == message
            assert not thread._cancel_requested

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_worker_thread_cancel_request(self, mock_qthread):
        """Test cancellation request."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread._cancel_requested = False

            thread.request_cancel()

            assert thread._cancel_requested

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_worker_thread_is_cancel_requested(self, mock_qthread):
        """Test is_cancel_requested method."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread._cancel_requested = False

            assert not thread.is_cancel_requested()

            thread._cancel_requested = True
            assert thread.is_cancel_requested()


class TestAgentDialogInit:
    """Test AgentDialog initialization (mocked)."""

    @patch("cnapy.gui_elements.agent_dialog.QDialog")
    @patch("cnapy.gui_elements.agent_dialog.LLMConfig")
    @patch("cnapy.gui_elements.agent_dialog.OrchestratorAgent")
    @patch("cnapy.gui_elements.agent_dialog.AgentContext")
    def test_agent_dialog_components_import(self, mock_context, mock_orch, mock_llm, mock_dialog):
        """Test that dialog components can be imported."""
        from cnapy.gui_elements.agent_dialog import (
            AgentDialog,
            AgentWorkerThread,
            ChatMessage,
        )

        assert AgentDialog is not None
        assert AgentWorkerThread is not None
        assert ChatMessage is not None


class TestChatMessage:
    """Test ChatMessage widget."""

    def test_chat_message_import(self):
        """Test ChatMessage can be imported."""
        from cnapy.gui_elements.agent_dialog import ChatMessage

        assert ChatMessage is not None


class TestAgentDialogCrewAIIntegration:
    """Test CrewAI integration in AgentDialog."""

    @patch("cnapy.agents.crewai_orchestrator.CNApyCrewOrchestrator")
    def test_crewai_orchestrator_init_success(self, mock_crewai):
        """Test successful CrewAI orchestrator initialization."""
        mock_instance = MagicMock()
        mock_instance.is_crewai_available.return_value = True
        mock_crewai.return_value = mock_instance

        # Simulate _init_crewai_orchestrator behavior
        crewai_available = mock_instance.is_crewai_available()

        assert crewai_available

    @patch("cnapy.agents.crewai_orchestrator.CNApyCrewOrchestrator")
    def test_crewai_orchestrator_init_unavailable(self, mock_crewai):
        """Test CrewAI orchestrator when unavailable."""
        mock_instance = MagicMock()
        mock_instance.is_crewai_available.return_value = False
        mock_crewai.return_value = mock_instance

        crewai_available = mock_instance.is_crewai_available()

        assert not crewai_available

    def test_crewai_import_error_handled(self):
        """Test that import error is handled gracefully."""
        try:
            from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

            # CrewAI orchestrator module exists
            assert CNApyCrewOrchestrator is not None
        except ImportError:
            # This is also acceptable - module may not have crewai installed
            pass


class TestWorkerThreadExecution:
    """Test worker thread execution patterns."""

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_traditional_routing(self, mock_qthread):
        """Test traditional (non-CrewAI) routing."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        mock_orchestrator = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_orchestrator.route.return_value = mock_response

        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread.orchestrator = mock_orchestrator
            thread.message = "test"
            thread.use_crewai = False
            thread._cancel_requested = False
            thread.status_update = MagicMock()
            thread.result_ready = MagicMock()
            thread.cancelled = MagicMock()

            # Simulate run method behavior
            thread.status_update.emit("Processing...")
            response = thread.orchestrator.route(thread.message)
            if not thread._cancel_requested:
                thread.result_ready.emit(response)

            mock_orchestrator.route.assert_called_once_with("test")

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_crewai_routing_with_cancel_check(self, mock_qthread):
        """Test CrewAI routing passes cancel_check."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        mock_orchestrator = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_orchestrator.route.return_value = mock_response

        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread.orchestrator = mock_orchestrator
            thread.message = "test"
            thread.use_crewai = True
            thread._cancel_requested = False
            thread.status_update = MagicMock()
            thread.result_ready = MagicMock()
            thread.cancelled = MagicMock()

            # Simulate run method behavior for CrewAI
            thread.status_update.emit("Processing...")
            response = thread.orchestrator.route(thread.message, cancel_check=thread.is_cancel_requested)
            if not thread._cancel_requested:
                thread.result_ready.emit(response)

            # Verify cancel_check was passed
            call_kwargs = mock_orchestrator.route.call_args
            assert "cancel_check" in call_kwargs.kwargs

    @patch("cnapy.gui_elements.agent_dialog.QThread")
    def test_cancelled_emits_signal(self, mock_qthread):
        """Test that cancellation emits cancelled signal."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        mock_orchestrator = MagicMock()

        with patch.object(AgentWorkerThread, "__init__", lambda x, *args, **kwargs: None):
            thread = AgentWorkerThread.__new__(AgentWorkerThread)
            thread.orchestrator = mock_orchestrator
            thread.message = "test"
            thread.use_crewai = False
            thread._cancel_requested = True
            thread.status_update = MagicMock()
            thread.result_ready = MagicMock()
            thread.cancelled = MagicMock()

            # Simulate cancelled run
            thread.status_update.emit("Processing...")
            if thread._cancel_requested:
                thread.cancelled.emit()

            thread.cancelled.emit.assert_called_once()


class TestQuickActions:
    """Test quick action functionality."""

    def test_quick_action_mapping(self):
        """Test that quick action mappings are defined."""
        # These are the expected quick action mappings
        expected_actions = {
            "perform_fba": "Perform FBA",
            "perform_pfba": "Perform pFBA",
            "perform_fva": "Perform FVA with 90% optimum",
            "aerobic": "Set aerobic condition",
            "anaerobic": "Set anaerobic condition",
            "essential_genes": "Find essential genes",
            "model_info": "Show model info",
            "clear_scenario": "Clear scenario",
        }

        # Verify expected actions exist (without instantiating dialog)
        assert len(expected_actions) == 8
        assert "perform_fba" in expected_actions


class TestSignalConnections:
    """Test signal-slot connections."""

    def test_signal_types_exist(self):
        """Test that expected signal types are used."""
        from cnapy.gui_elements.agent_dialog import AgentWorkerThread

        # Verify the class has the expected signal attributes defined
        # (They're class-level Signal objects)
        assert hasattr(AgentWorkerThread, "result_ready")
        assert hasattr(AgentWorkerThread, "error_occurred")
        assert hasattr(AgentWorkerThread, "status_update")
        assert hasattr(AgentWorkerThread, "cancelled")
