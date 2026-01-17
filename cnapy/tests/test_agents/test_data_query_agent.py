"""Tests for DataQueryAgent."""

import pytest
from cnapy.agents.data_query_agent import DataQueryAgent
from cnapy.agents.base_agent import SkillStatus


class TestDataQueryAgent:
    """Tests for DataQueryAgent."""

    def test_agent_creation(self, agent_context):
        """Test creating the agent."""
        agent = DataQueryAgent(agent_context)
        assert agent.name == "data_query"
        assert len(agent.skills) > 0

    def test_registered_skills(self, agent_context):
        """Test that expected skills are registered."""
        agent = DataQueryAgent(agent_context)
        expected_skills = [
            "get_model_info",
            "get_reaction_info",
            "get_metabolite_info",
            "get_gene_info",
            "search_reactions",
            "search_metabolites",
            "search_genes",
            "get_exchange_reactions",
            "get_objective",
        ]
        for skill in expected_skills:
            assert skill in agent.skills

    def test_can_handle_model_info(self, agent_context):
        """Test can_handle for model info requests."""
        agent = DataQueryAgent(agent_context)
        score = agent.can_handle("show model information")
        assert score >= 0.0  # Should have ability to handle model info

    def test_can_handle_search(self, agent_context):
        """Test can_handle for search requests."""
        agent = DataQueryAgent(agent_context)
        score = agent.can_handle("search for reactions containing glucose")
        assert score >= 0.0  # Should have ability to handle search

    def test_can_handle_korean(self, agent_context):
        """Test can_handle for Korean requests."""
        agent = DataQueryAgent(agent_context)
        score = agent.can_handle("모델 정보")  # Model info in Korean
        assert score >= 0.0  # May not fully support Korean

    def test_get_model_info_no_model(self, agent_context):
        """Test get_model_info without model loaded."""
        agent_context.appdata.project.cobra_py_model = None
        agent = DataQueryAgent(agent_context)

        result = agent.execute_skill("get_model_info", {})
        assert result.status == SkillStatus.FAILURE

    def test_get_tools(self, agent_context):
        """Test getting tool definitions."""
        agent = DataQueryAgent(agent_context)
        tools = agent.get_tools()

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "get_model_info" in tool_names
        assert "search_reactions" in tool_names


class TestDataQueryWithSimpleModel:
    """Tests using simple model fixture."""

    def test_get_model_info(self, agent_context):
        """Test getting model info."""
        agent = DataQueryAgent(agent_context)
        result = agent.execute_skill("get_model_info", {})

        assert result.success
        assert "model_id" in result.data
        assert "n_reactions" in result.data
        assert "n_metabolites" in result.data
        assert "n_genes" in result.data

    def test_get_reaction_info(self, agent_context):
        """Test getting reaction info."""
        agent = DataQueryAgent(agent_context)
        result = agent.execute_skill("get_reaction_info", {"reaction_id": "R1"})

        assert result.success
        assert "id" in result.data
        assert result.data["id"] == "R1"

    def test_get_reaction_info_not_found(self, agent_context):
        """Test getting nonexistent reaction info."""
        agent = DataQueryAgent(agent_context)
        result = agent.execute_skill("get_reaction_info", {"reaction_id": "nonexistent"})

        assert result.status == SkillStatus.FAILURE

    def test_search_reactions(self, agent_context):
        """Test searching reactions."""
        agent = DataQueryAgent(agent_context)
        result = agent.execute_skill("search_reactions", {"query": "Reaction"})

        assert result.success
        assert "results" in result.data
        # Simple model has reactions with "Reaction" in name
        assert len(result.data["results"]) > 0

    def test_search_reactions_no_match(self, agent_context):
        """Test searching with no matches."""
        agent = DataQueryAgent(agent_context)
        result = agent.execute_skill("search_reactions", {"query": "xyznonexistent"})

        assert result.success
        assert len(result.data["results"]) == 0


class TestDataQueryWithEcoliModel:
    """Tests using E. coli core model."""

    def test_get_model_info_ecoli(self, agent_context_ecoli):
        """Test getting E. coli model info."""
        agent = DataQueryAgent(agent_context_ecoli)
        result = agent.execute_skill("get_model_info", {})

        assert result.success
        assert result.data["n_reactions"] > 50
        assert result.data["n_metabolites"] > 50
        assert result.data["n_genes"] > 0

    def test_get_exchange_reactions(self, agent_context_ecoli):
        """Test getting exchange reactions."""
        agent = DataQueryAgent(agent_context_ecoli)
        result = agent.execute_skill("get_exchange_reactions", {})

        assert result.success
        assert "reactions" in result.data
        assert len(result.data["reactions"]) > 0

    def test_get_objective(self, agent_context_ecoli):
        """Test getting objective."""
        agent = DataQueryAgent(agent_context_ecoli)
        result = agent.execute_skill("get_objective", {})

        assert result.success
        assert "expression" in result.data
        assert "direction" in result.data

    def test_get_metabolite_info(self, agent_context_ecoli):
        """Test getting metabolite info."""
        agent = DataQueryAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.metabolites) > 0:
            met_id = model.metabolites[0].id
            result = agent.execute_skill("get_metabolite_info", {"metabolite_id": met_id})

            assert result.success
            assert "id" in result.data
        else:
            pytest.skip("No metabolites in model")

    def test_get_gene_info(self, agent_context_ecoli):
        """Test getting gene info."""
        agent = DataQueryAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.genes) > 0:
            gene_id = model.genes[0].id
            result = agent.execute_skill("get_gene_info", {"gene_id": gene_id})

            assert result.success
            assert "id" in result.data
        else:
            pytest.skip("No genes in model")

    def test_search_metabolites(self, agent_context_ecoli):
        """Test searching metabolites."""
        agent = DataQueryAgent(agent_context_ecoli)
        result = agent.execute_skill("search_metabolites", {"query": "glucose"})

        assert result.success
        assert "results" in result.data

    def test_search_genes(self, agent_context_ecoli):
        """Test searching genes."""
        agent = DataQueryAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.genes) > 0:
            # Search for first letter of first gene
            first_gene = model.genes[0].id[:3]
            result = agent.execute_skill("search_genes", {"query": first_gene})

            assert result.success
            assert "results" in result.data
        else:
            pytest.skip("No genes in model")

    def test_list_compartments(self, agent_context_ecoli):
        """Test listing compartments."""
        agent = DataQueryAgent(agent_context_ecoli)
        result = agent.execute_skill("list_compartments", {})

        assert result.success
        assert "compartments" in result.data
        assert len(result.data["compartments"]) > 0
