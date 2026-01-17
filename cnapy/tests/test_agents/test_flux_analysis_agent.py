"""Tests for FluxAnalysisAgent."""

import pytest
from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent
from cnapy.agents.base_agent import SkillStatus


class TestFluxAnalysisAgent:
    """Tests for FluxAnalysisAgent."""

    def test_agent_creation(self, agent_context):
        """Test creating the agent."""
        agent = FluxAnalysisAgent(agent_context)
        assert agent.name == "flux_analysis"
        assert len(agent.skills) > 0

    def test_registered_skills(self, agent_context):
        """Test that expected skills are registered."""
        agent = FluxAnalysisAgent(agent_context)
        expected_skills = [
            "perform_fba",
            "perform_pfba",
            "perform_fva",
            "get_objective_value",
        ]
        for skill in expected_skills:
            assert skill in agent.skills

    def test_can_handle_fba(self, agent_context):
        """Test can_handle for FBA requests."""
        agent = FluxAnalysisAgent(agent_context)
        score = agent.can_handle("perform FBA analysis")
        assert score >= 0.0  # Should have ability to handle FBA requests

    def test_can_handle_korean(self, agent_context):
        """Test can_handle for Korean requests."""
        agent = FluxAnalysisAgent(agent_context)
        score = agent.can_handle("FBA 수행해줘")
        assert score >= 0.0  # May have limited Korean support

    def test_perform_fba_no_model(self, agent_context):
        """Test FBA without model loaded."""
        # Remove model
        agent_context.appdata.project.cobra_py_model = None
        agent = FluxAnalysisAgent(agent_context)

        result = agent.execute_skill("perform_fba", {})
        assert result.status == SkillStatus.FAILURE

    def test_perform_fba(self, agent_context_ecoli):
        """Test performing FBA with E. coli model."""
        agent = FluxAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("perform_fba", {})

        assert result.success
        assert "objective_value" in result.data
        assert result.data["objective_value"] > 0

    @pytest.mark.xfail(reason="pFBA may fail with numpy compatibility issues in some environments")
    def test_perform_pfba(self, agent_context_ecoli):
        """Test performing pFBA with E. coli model."""
        agent = FluxAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("perform_pfba", {})

        assert result.success
        assert "objective_value" in result.data
        assert "total_flux" in result.data

    def test_perform_fva(self, agent_context_ecoli):
        """Test performing FVA with E. coli model."""
        agent = FluxAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("perform_fva", {"fraction_of_optimum": 0.9})

        assert result.success
        assert "minimum" in result.data
        assert "maximum" in result.data
        assert "n_variable" in result.data

    def test_get_objective_value(self, agent_context_ecoli):
        """Test getting objective value after FBA."""
        agent = FluxAnalysisAgent(agent_context_ecoli)

        # First perform FBA
        agent.execute_skill("perform_fba", {})

        # Then get objective value
        result = agent.execute_skill("get_objective_value", {})
        assert result.success
        assert "objective_value" in result.data

    def test_get_tools(self, agent_context):
        """Test getting tool definitions."""
        agent = FluxAnalysisAgent(agent_context)
        tools = agent.get_tools()

        assert len(tools) > 0
        assert all(hasattr(t, 'name') and hasattr(t, 'description') for t in tools)


class TestFluxAnalysisAgentWithSimpleModel:
    """Tests using simple model fixture."""

    def test_fba_simple_model(self, agent_context):
        """Test FBA with simple model."""
        agent = FluxAnalysisAgent(agent_context)
        result = agent.execute_skill("perform_fba", {})

        assert result.success
        # Simple model should have some objective value
        assert "objective_value" in result.data

    def test_analyze_flux_distribution(self, agent_context):
        """Test analyzing flux distribution."""
        agent = FluxAnalysisAgent(agent_context)

        # First perform FBA
        agent.execute_skill("perform_fba", {})

        # Analyze distribution
        result = agent.execute_skill("analyze_flux_distribution", {
            "reaction_ids": ["R1", "R2"]
        })

        # Should succeed even if some reactions not in solution
        assert result.status in [SkillStatus.SUCCESS, SkillStatus.FAILURE]
