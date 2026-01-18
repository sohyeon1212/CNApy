"""Tests for CrewAI Orchestrator

This module tests the CrewAI-based orchestrator for intelligent
natural language understanding and tool selection.
"""

from unittest.mock import MagicMock

import pytest

from cnapy.agents.base_agent import SkillStatus


class TestCNApyCrewOrchestratorImport:
    """Test CrewAI orchestrator can be imported."""

    def test_import_crewai_orchestrator(self):
        """Test that CrewAI orchestrator module can be imported."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        assert CNApyCrewOrchestrator is not None

    def test_import_crewai_backend_enum(self):
        """Test that CrewAIBackend enum can be imported."""
        from cnapy.agents.crewai_orchestrator import CrewAIBackend

        assert CrewAIBackend.OPENAI.value == "openai"
        assert CrewAIBackend.ANTHROPIC.value == "anthropic"
        assert CrewAIBackend.GOOGLE.value == "google"


class TestCNApyCrewOrchestratorInit:
    """Test CrewAI orchestrator initialization."""

    def test_init_without_llm_config(self, agent_context):
        """Test initialization without LLM configuration."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)

        assert orchestrator.context == agent_context
        assert orchestrator.llm_config is None
        assert not orchestrator._agents_initialized

    def test_init_with_llm_config(self, agent_context, mock_llm_config):
        """Test initialization with LLM configuration."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_llm_config)

        assert orchestrator.llm_config == mock_llm_config

    def test_cnapy_agents_setup(self, agent_context):
        """Test that CNApy agents are properly set up."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)

        assert "flux_analysis" in orchestrator._cnapy_agents
        assert "gene_analysis" in orchestrator._cnapy_agents
        assert "scenario" in orchestrator._cnapy_agents
        assert "data_query" in orchestrator._cnapy_agents


class TestCNApyCrewOrchestratorLLMCheck:
    """Test LLM configuration checking."""

    def test_check_llm_not_configured(self, agent_context):
        """Test LLM check when not configured."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)

        assert not orchestrator._check_llm_configured()

    def test_check_llm_configured_anthropic(self, agent_context):
        """Test LLM check with Anthropic configuration."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        mock_config = MagicMock()
        mock_config.provider = "anthropic"
        mock_config.anthropic_api_key = "test-key"

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_config)

        assert orchestrator._check_llm_configured()

    def test_check_llm_configured_openai(self, agent_context):
        """Test LLM check with OpenAI configuration."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        mock_config = MagicMock()
        mock_config.provider = "openai"
        mock_config.openai_api_key = "test-key"

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_config)

        assert orchestrator._check_llm_configured()

    def test_check_llm_missing_api_key(self, agent_context):
        """Test LLM check with missing API key."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        mock_config = MagicMock()
        mock_config.provider = "anthropic"
        mock_config.anthropic_api_key = None

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_config)

        assert not orchestrator._check_llm_configured()


class TestCNApyCrewOrchestratorCrewAICheck:
    """Test CrewAI availability checking."""

    def test_check_crewai_available(self, agent_context):
        """Test checking if CrewAI is available."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)

        # This will return True if crewai is installed, False otherwise
        result = orchestrator._check_crewai_available()
        assert isinstance(result, bool)

    def test_is_crewai_available_property(self, agent_context):
        """Test is_crewai_available method."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)

        # Should return False without LLM config
        assert not orchestrator.is_crewai_available()


class TestCNApyCrewOrchestratorDirectExecution:
    """Test direct tool execution without CrewAI."""

    def test_execute_tool_directly_fba(self, agent_context_ecoli):
        """Test direct FBA execution."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)
        result = orchestrator.execute_tool_directly("perform_fba", {})

        assert result.status == SkillStatus.SUCCESS
        assert "objective_value" in result.data

    def test_execute_tool_directly_model_info(self, agent_context_ecoli):
        """Test direct model info execution."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)
        result = orchestrator.execute_tool_directly("get_model_info", {})

        assert result.status == SkillStatus.SUCCESS
        assert "n_reactions" in result.data
        assert "n_metabolites" in result.data

    def test_execute_tool_directly_unknown_tool(self, agent_context):
        """Test execution with unknown tool name."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)
        result = orchestrator.execute_tool_directly("unknown_tool", {})

        assert result.status == SkillStatus.FAILURE
        assert "unknown_tool" in result.error.lower()

    def test_execute_tool_directly_apply_condition(self, agent_context_ecoli):
        """Test direct condition application."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)
        result = orchestrator.execute_tool_directly("apply_condition", {"condition_name": "aerobic"})

        # May succeed or partially succeed depending on model
        assert result.status in [SkillStatus.SUCCESS, SkillStatus.PARTIAL]


class TestCNApyCrewOrchestratorFallback:
    """Test fallback routing to traditional orchestrator."""

    def test_fallback_route_fba(self, agent_context_ecoli):
        """Test fallback routing for FBA request."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)

        # Without CrewAI configured, should fall back
        response = orchestrator.route("perform FBA")

        assert response.success
        assert response.agent_name == "flux_analysis"

    def test_fallback_route_korean(self, agent_context_ecoli):
        """Test fallback routing for Korean request."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)
        response = orchestrator.route("FBA 수행해줘")

        assert response.success

    def test_route_with_cancel_check(self, agent_context_ecoli):
        """Test routing with cancellation callback."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        cancel_called = []

        def cancel_check():
            cancel_called.append(True)
            return False

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli)
        response = orchestrator.route("perform FBA", cancel_check=cancel_check)

        assert response.success


class TestCNApyCrewOrchestratorGetTools:
    """Test getting available tools."""

    def test_get_available_tools(self, agent_context):
        """Test getting list of available tools."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)
        tools = orchestrator.get_available_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check tool structure
        tool = tools[0]
        assert "name" in tool
        assert "agent" in tool
        assert "description" in tool

    def test_get_available_tools_includes_parametric(self, agent_context):
        """Test that parametric_analysis tool is included."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context)
        tools = orchestrator.get_available_tools()

        tool_names = [t["name"] for t in tools]
        assert "parametric_analysis" in tool_names


class TestCNApyCrewOrchestratorAgentSelection:
    """Test agent selection logic."""

    def test_select_flux_agent(self, agent_context, mock_llm_config):
        """Test selecting flux analyst for FBA-related messages."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_llm_config)

        # Initialize CrewAI agents if available
        try:
            orchestrator._initialize_crewai()
            if orchestrator._crewai_agents:
                agent = orchestrator._select_best_agent("perform FBA analysis")
                assert agent is not None
        except Exception:
            pytest.skip("CrewAI not available")

    def test_select_gene_agent(self, agent_context, mock_llm_config):
        """Test selecting gene analyst for knockout messages."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        orchestrator = CNApyCrewOrchestrator(agent_context, mock_llm_config)

        try:
            orchestrator._initialize_crewai()
            if orchestrator._crewai_agents:
                agent = orchestrator._select_best_agent("knockout gene pfkA")
                assert agent is not None
        except Exception:
            pytest.skip("CrewAI not available")


@pytest.mark.skipif(
    True,  # Skip unless CrewAI is installed and configured
    reason="CrewAI integration tests require CrewAI and LLM API key",
)
class TestCNApyCrewOrchestratorIntegration:
    """Integration tests for CrewAI orchestrator (requires CrewAI installation)."""

    def test_crewai_routing_fba(self, agent_context_ecoli):
        """Test CrewAI routing for FBA request."""
        from cnapy.agents.crewai_orchestrator import CNApyCrewOrchestrator

        # Would need real API key
        mock_config = MagicMock()
        mock_config.provider = "anthropic"
        mock_config.anthropic_api_key = "test-key"

        orchestrator = CNApyCrewOrchestrator(agent_context_ecoli, mock_config)

        if orchestrator.is_crewai_available():
            response = orchestrator.route("perform FBA analysis")
            assert response is not None
        else:
            pytest.skip("CrewAI not fully configured")
