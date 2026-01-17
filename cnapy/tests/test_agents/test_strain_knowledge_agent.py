"""Tests for StrainKnowledgeAgent."""

import pytest
from unittest.mock import MagicMock, patch
from cnapy.agents.strain_knowledge_agent import StrainKnowledgeAgent
from cnapy.agents.base_agent import SkillStatus


class TestStrainKnowledgeAgent:
    """Tests for StrainKnowledgeAgent."""

    def test_agent_creation(self, agent_context, mock_llm_config):
        """Test creating the agent."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        assert agent.name == "StrainKnowledgeAgent"
        assert len(agent.skills) > 0

    def test_registered_skills(self, agent_context, mock_llm_config):
        """Test that expected skills are registered."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        expected_skills = [
            "analyze_strain_reactions",
            "analyze_strain_genes",
            "get_strain_metabolism",
            "compare_strains",
            "suggest_modifications",
            "literature_search",
            "check_reaction_in_strain",
            "check_gene_in_strain",
        ]
        for skill in expected_skills:
            assert skill in agent.skills

    def test_can_handle_strain_query(self, agent_context, mock_llm_config):
        """Test can_handle for strain-related requests."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        score = agent.can_handle("does this reaction exist in E. coli")
        assert score > 0.3

    def test_can_handle_korean(self, agent_context, mock_llm_config):
        """Test can_handle for Korean requests."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        # Use pattern that matches our TRIGGER_KEYWORDS
        score = agent.can_handle("균주에서 반응 존재 여부")
        assert score > 0.3

    def test_can_handle_ortholog(self, agent_context, mock_llm_config):
        """Test can_handle for ortholog requests."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        score = agent.can_handle("find ortholog of pgi in yeast")
        assert score > 0.3

    def test_get_tools(self, agent_context, mock_llm_config):
        """Test getting tool definitions."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        tools = agent.get_tools()

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "analyze_strain_reactions" in tool_names
        assert "compare_strains" in tool_names


class TestStrainKnowledgeAgentSkills:
    """Tests for individual skills with mocked LLM."""

    @pytest.fixture
    def agent_with_mock_llm(self, agent_context, mock_llm_config):
        """Create agent with mocked LLM calls."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)
        return agent

    def test_analyze_strain_reactions_no_strain(self, agent_with_mock_llm):
        """Test analyze_strain_reactions without strain name."""
        result = agent_with_mock_llm.execute_skill("analyze_strain_reactions", {})
        # Required param missing = FAILURE
        assert result.status == SkillStatus.FAILURE

    def test_analyze_strain_genes_no_strain(self, agent_with_mock_llm):
        """Test analyze_strain_genes without strain name."""
        result = agent_with_mock_llm.execute_skill("analyze_strain_genes", {})
        assert result.status == SkillStatus.FAILURE

    def test_get_strain_metabolism_no_strain(self, agent_with_mock_llm):
        """Test get_strain_metabolism without strain name."""
        result = agent_with_mock_llm.execute_skill("get_strain_metabolism", {})
        assert result.status == SkillStatus.FAILURE

    def test_compare_strains_missing_params(self, agent_with_mock_llm):
        """Test compare_strains with missing parameters."""
        result = agent_with_mock_llm.execute_skill("compare_strains", {"strain1": "E. coli"})
        assert result.status == SkillStatus.FAILURE

    def test_suggest_modifications_missing_params(self, agent_with_mock_llm):
        """Test suggest_modifications with missing parameters."""
        result = agent_with_mock_llm.execute_skill("suggest_modifications", {})
        assert result.status == SkillStatus.FAILURE

    def test_literature_search_no_query(self, agent_with_mock_llm):
        """Test literature_search without query."""
        result = agent_with_mock_llm.execute_skill("literature_search", {})
        assert result.status == SkillStatus.FAILURE

    def test_check_reaction_in_strain_missing_params(self, agent_with_mock_llm):
        """Test check_reaction_in_strain with missing parameters."""
        result = agent_with_mock_llm.execute_skill("check_reaction_in_strain", {})
        assert result.status == SkillStatus.FAILURE

    def test_check_gene_in_strain_missing_params(self, agent_with_mock_llm):
        """Test check_gene_in_strain with missing parameters."""
        result = agent_with_mock_llm.execute_skill("check_gene_in_strain", {})
        assert result.status == SkillStatus.FAILURE


class TestLLMIntegration:
    """Tests for LLM integration (with mocked API calls)."""

    @pytest.fixture
    def mock_llm_response(self):
        """Return mock LLM response."""
        return """{
            "exists": "Yes",
            "confidence": "High",
            "evidence": "This is a well-characterized enzyme in E. coli",
            "references": ["ref1", "ref2"]
        }"""

    def test_analyze_strain_reactions_mocked(self, agent_context, mock_llm_config, mock_llm_response):
        """Test analyze_strain_reactions with mocked LLM."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        with patch.object(agent, '_call_llm', return_value=mock_llm_response):
            result = agent.execute_skill("analyze_strain_reactions", {
                "strain_name": "E. coli",
                "reaction_ids": ["PGI", "PFK"]
            })

            assert result.success
            assert result.data is not None

    def test_get_strain_metabolism_mocked(self, agent_context, mock_llm_config):
        """Test get_strain_metabolism with mocked LLM."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        mock_response = """{
            "strain": "E. coli",
            "characteristics": {
                "key_features": ["feature1", "feature2"],
                "pathways": {"present": ["glycolysis"], "absent": []}
            },
            "summary": "E. coli is a well-studied bacterium"
        }"""

        with patch.object(agent, '_call_llm', return_value=mock_response):
            result = agent.execute_skill("get_strain_metabolism", {
                "strain_name": "E. coli",
                "aspect": "general"
            })

            assert result.success
            assert result.data is not None

    def test_compare_strains_mocked(self, agent_context, mock_llm_config):
        """Test compare_strains with mocked LLM."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        mock_response = """{
            "strain1": "E. coli",
            "strain2": "C. glutamicum",
            "comparison": {
                "similarities": ["both are bacteria"],
                "differences": []
            },
            "summary": "Comparison summary"
        }"""

        with patch.object(agent, '_call_llm', return_value=mock_response):
            result = agent.execute_skill("compare_strains", {
                "strain1": "E. coli",
                "strain2": "C. glutamicum"
            })

            assert result.success
            assert result.data is not None

    def test_suggest_modifications_mocked(self, agent_context, mock_llm_config):
        """Test suggest_modifications with mocked LLM."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        mock_response = """{
            "strain": "E. coli",
            "target_product": "lysine",
            "strategies": {
                "knockouts": [],
                "overexpressions": []
            },
            "summary": "Strategy summary"
        }"""

        with patch.object(agent, '_call_llm', return_value=mock_response):
            result = agent.execute_skill("suggest_modifications", {
                "strain_name": "E. coli",
                "target_product": "lysine"
            })

            assert result.success
            assert result.data is not None


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_code_block(self, agent_context, mock_llm_config):
        """Test parsing JSON from code block."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        response = """Here is the analysis:

```json
{
    "exists": "Yes",
    "confidence": "High"
}
```
"""
        result = agent._parse_json_response(response)
        assert result["exists"] == "Yes"
        assert result["confidence"] == "High"

    def test_parse_json_inline(self, agent_context, mock_llm_config):
        """Test parsing inline JSON."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        response = 'The result is {"exists": "No", "confidence": "Low"}'
        result = agent._parse_json_response(response)
        assert result["exists"] == "No"
        assert result["confidence"] == "Low"

    def test_parse_json_nested(self, agent_context, mock_llm_config):
        """Test parsing nested JSON."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        response = """
{
    "outer": {
        "inner": {"value": 42}
    },
    "array": [1, 2, 3]
}
"""
        result = agent._parse_json_response(response)
        assert result["outer"]["inner"]["value"] == 42
        assert result["array"] == [1, 2, 3]

    def test_parse_json_fallback(self, agent_context, mock_llm_config):
        """Test fallback when JSON parsing fails."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        response = "This is not JSON at all"
        result = agent._parse_json_response(response)
        assert "raw_response" in result
        assert result["parsed"] is False


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_key_generation(self, agent_context, mock_llm_config):
        """Test that cache keys are generated consistently."""
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        key1 = agent._get_cache_key("reaction", "E. coli", "PGI")
        key2 = agent._get_cache_key("reaction", "E. coli", "PGI")
        key3 = agent._get_cache_key("reaction", "E. coli", "PFK")

        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different key

    def test_cache_disabled(self, agent_context, mock_llm_config):
        """Test that cache can be disabled."""
        mock_llm_config.use_cache = False
        agent = StrainKnowledgeAgent(agent_context, mock_llm_config)

        result = agent._get_cached_result("some_key")
        assert result is None
