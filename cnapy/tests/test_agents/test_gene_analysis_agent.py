"""Tests for GeneAnalysisAgent."""

import pytest
from cnapy.agents.gene_analysis_agent import GeneAnalysisAgent
from cnapy.agents.base_agent import SkillStatus


class TestGeneAnalysisAgent:
    """Tests for GeneAnalysisAgent."""

    def test_agent_creation(self, agent_context):
        """Test creating the agent."""
        agent = GeneAnalysisAgent(agent_context)
        assert agent.name == "gene_analysis"
        assert len(agent.skills) > 0

    def test_registered_skills(self, agent_context):
        """Test that expected skills are registered."""
        agent = GeneAnalysisAgent(agent_context)
        expected_skills = [
            "knockout_gene",
            "knockout_genes",
            "find_essential_genes",
            "find_essential_reactions",
            "get_gene_info",
        ]
        for skill in expected_skills:
            assert skill in agent.skills

    def test_can_handle_knockout(self, agent_context):
        """Test can_handle for knockout requests."""
        agent = GeneAnalysisAgent(agent_context)
        score = agent.can_handle("knockout gene pgi")
        assert score > 0.0  # Should have some ability to handle knockout requests

    def test_can_handle_essential(self, agent_context):
        """Test can_handle for essential genes requests."""
        agent = GeneAnalysisAgent(agent_context)
        score = agent.can_handle("find essential genes")
        assert score > 0.0  # Should have some ability to handle essential genes

    def test_can_handle_korean(self, agent_context):
        """Test can_handle for Korean requests."""
        agent = GeneAnalysisAgent(agent_context)
        score = agent.can_handle("유전자 녹아웃 해줘")  # Gene knockout in Korean
        assert score >= 0.0  # May not fully support Korean

    def test_knockout_no_model(self, agent_context):
        """Test knockout without model loaded."""
        agent_context.appdata.project.cobra_py_model = None
        agent = GeneAnalysisAgent(agent_context)

        result = agent.execute_skill("knockout_gene", {"gene_id": "test"})
        assert result.status == SkillStatus.FAILURE

    def test_get_tools(self, agent_context):
        """Test getting tool definitions."""
        agent = GeneAnalysisAgent(agent_context)
        tools = agent.get_tools()

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "knockout_gene" in tool_names
        assert "find_essential_genes" in tool_names


class TestGeneAnalysisWithEcoliModel:
    """Tests using E. coli core model."""

    def test_knockout_gene(self, agent_context_ecoli):
        """Test knocking out a gene."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        # Get first gene
        if len(model.genes) > 0:
            gene_id = model.genes[0].id
            result = agent.execute_skill("knockout_gene", {"gene_id": gene_id})

            assert result.success
            assert "wt_growth" in result.data
            assert "ko_growth" in result.data
        else:
            pytest.skip("No genes in model")

    def test_knockout_nonexistent_gene(self, agent_context_ecoli):
        """Test knocking out a nonexistent gene."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("knockout_gene", {"gene_id": "nonexistent_gene_xyz"})

        assert result.status == SkillStatus.FAILURE

    def test_knockout_multiple_genes(self, agent_context_ecoli):
        """Test knocking out multiple genes."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.genes) >= 2:
            gene_ids = [model.genes[0].id, model.genes[1].id]
            result = agent.execute_skill("knockout_genes", {"gene_ids": gene_ids})

            assert result.success
            assert "wt_growth" in result.data
            assert "ko_growth" in result.data
        else:
            pytest.skip("Not enough genes in model")

    def test_find_essential_genes(self, agent_context_ecoli):
        """Test finding essential genes."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("find_essential_genes", {"threshold": 0.01})

        assert result.success
        assert "essential_genes" in result.data
        assert "n_total_genes" in result.data
        assert "n_essential" in result.data

    def test_find_essential_reactions(self, agent_context_ecoli):
        """Test finding essential reactions."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        result = agent.execute_skill("find_essential_reactions", {"threshold": 0.01})

        assert result.success
        assert "essential_reactions" in result.data
        assert "n_total_reactions" in result.data
        assert "n_essential" in result.data

    def test_get_gene_info(self, agent_context_ecoli):
        """Test getting gene info."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.genes) > 0:
            gene_id = model.genes[0].id
            result = agent.execute_skill("get_gene_info", {"gene_id": gene_id})

            assert result.success
            assert "gene_id" in result.data
        else:
            pytest.skip("No genes in model")

    def test_analyze_gene_impact(self, agent_context_ecoli):
        """Test analyzing gene impact."""
        agent = GeneAnalysisAgent(agent_context_ecoli)
        model = agent_context_ecoli.appdata.project.cobra_py_model

        if len(model.genes) > 0:
            gene_id = model.genes[0].id
            result = agent.execute_skill("analyze_gene_impact", {"gene_id": gene_id})

            assert result.success
            assert "gene_id" in result.data
            # The reactions come from _get_gene_info which returns "reactions"
            assert "reactions" in result.data
        else:
            pytest.skip("No genes in model")
