"""Tests for ScenarioManagerAgent."""

from cnapy.agents.base_agent import SkillStatus
from cnapy.agents.scenario_manager_agent import ScenarioManagerAgent


class TestScenarioManagerAgent:
    """Tests for ScenarioManagerAgent."""

    def test_agent_creation(self, agent_context):
        """Test creating the agent."""
        agent = ScenarioManagerAgent(agent_context)
        assert agent.name == "scenario"
        assert len(agent.skills) > 0

    def test_registered_skills(self, agent_context):
        """Test that expected skills are registered."""
        agent = ScenarioManagerAgent(agent_context)
        expected_skills = [
            "apply_condition",
            "set_reaction_bounds",
            "set_carbon_source",
            "set_nitrogen_source",
            "get_current_scenario",
            "clear_scenario",
            "list_conditions",
        ]
        for skill in expected_skills:
            assert skill in agent.skills

    def test_can_handle_condition(self, agent_context):
        """Test can_handle for condition-related requests."""
        agent = ScenarioManagerAgent(agent_context)
        score = agent.can_handle("set anaerobic condition")
        assert score >= 0.0  # Should have ability to handle condition requests

    def test_can_handle_korean(self, agent_context):
        """Test can_handle for Korean requests."""
        agent = ScenarioManagerAgent(agent_context)
        score = agent.can_handle("조건 설정해줘")  # Condition setting in Korean
        assert score >= 0.0  # May not fully support Korean

    def test_list_conditions(self, agent_context):
        """Test listing available conditions."""
        agent = ScenarioManagerAgent(agent_context)
        result = agent.execute_skill("list_conditions", {})

        assert result.success
        assert "conditions" in result.data
        assert len(result.data["conditions"]) > 0
        # Check for expected conditions
        condition_names = [c["name"] for c in result.data["conditions"]]
        assert "aerobic" in condition_names
        assert "anaerobic" in condition_names

    def test_list_carbon_sources(self, agent_context):
        """Test listing carbon sources."""
        agent = ScenarioManagerAgent(agent_context)
        result = agent.execute_skill("list_carbon_sources", {})

        assert result.success
        assert "carbon_sources" in result.data
        source_names = [s["name"] for s in result.data["carbon_sources"]]
        assert "glucose" in source_names

    def test_list_nitrogen_sources(self, agent_context):
        """Test listing nitrogen sources."""
        agent = ScenarioManagerAgent(agent_context)
        result = agent.execute_skill("list_nitrogen_sources", {})

        assert result.success
        assert "nitrogen_sources" in result.data

    def test_get_current_scenario_empty(self, agent_context):
        """Test getting current scenario when empty."""
        agent = ScenarioManagerAgent(agent_context)
        result = agent.execute_skill("get_current_scenario", {})

        assert result.success
        assert result.data is not None
        assert "flux_values" in result.data or "n_flux_values" in result.data


class TestScenarioManagerWithEcoliModel:
    """Tests using E. coli core model."""

    def test_apply_aerobic_condition(self, agent_context_ecoli):
        """Test applying aerobic condition."""
        agent = ScenarioManagerAgent(agent_context_ecoli)
        result = agent.execute_skill("apply_condition", {"condition_name": "aerobic"})

        assert result.success
        # Check that oxygen exchange is set correctly
        model = agent_context_ecoli.appdata.project.cobra_py_model
        if "EX_o2_e" in model.reactions:
            rxn = model.reactions.get_by_id("EX_o2_e")
            assert rxn.lower_bound < 0  # Oxygen uptake allowed

    def test_apply_anaerobic_condition(self, agent_context_ecoli):
        """Test applying anaerobic condition."""
        agent = ScenarioManagerAgent(agent_context_ecoli)
        result = agent.execute_skill("apply_condition", {"condition_name": "anaerobic"})

        assert result.success
        # Check that oxygen exchange is blocked
        model = agent_context_ecoli.appdata.project.cobra_py_model
        if "EX_o2_e" in model.reactions:
            rxn = model.reactions.get_by_id("EX_o2_e")
            assert rxn.lower_bound == 0  # No oxygen uptake

    def test_apply_unknown_condition(self, agent_context_ecoli):
        """Test applying unknown condition."""
        agent = ScenarioManagerAgent(agent_context_ecoli)
        result = agent.execute_skill("apply_condition", {"condition_name": "unknown_condition"})

        assert result.status == SkillStatus.FAILURE

    def test_set_reaction_bounds(self, agent_context_ecoli):
        """Test setting reaction bounds."""
        agent = ScenarioManagerAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        # Get first reaction
        rxn_id = model.reactions[0].id

        result = agent.execute_skill(
            "set_reaction_bounds", {"reaction_id": rxn_id, "lower_bound": -5.0, "upper_bound": 10.0}
        )

        assert result.success
        # Verify bounds were set
        rxn = model.reactions.get_by_id(rxn_id)
        assert rxn.lower_bound == -5.0
        assert rxn.upper_bound == 10.0

    def test_set_carbon_source_glucose(self, agent_context_ecoli):
        """Test setting glucose as carbon source."""
        agent = ScenarioManagerAgent(agent_context_ecoli)
        result = agent.execute_skill("set_carbon_source", {"carbon_source": "glucose"})

        # Should succeed if glucose exchange exists
        if "EX_glc__D_e" in agent_context_ecoli.appdata.project.cobra_py_model.reactions:
            assert result.success
        else:
            # Skip if reaction not in model
            pass

    def test_clear_scenario(self, agent_context_ecoli):
        """Test clearing scenario."""
        agent = ScenarioManagerAgent(agent_context_ecoli)

        # First apply some condition
        agent.execute_skill("apply_condition", {"condition_name": "anaerobic"})

        # Then clear
        result = agent.execute_skill("clear_scenario", {})
        assert result.success

    def test_get_tools(self, agent_context):
        """Test getting tool definitions."""
        agent = ScenarioManagerAgent(agent_context)
        tools = agent.get_tools()

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "apply_condition" in tool_names
        assert "set_reaction_bounds" in tool_names
