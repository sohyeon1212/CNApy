"""Tests for AgentRegistry."""

import pytest
from cnapy.agents.agent_registry import (
    AgentRegistry, ROUTING_RULES, create_default_registry
)
from cnapy.agents.base_agent import BaseAgent, AgentContext, SkillResult, SkillStatus


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, context: AgentContext):
        super().__init__(context)

    @property
    def name(self) -> str:
        return "mock_agent"

    @property
    def description(self) -> str:
        return "Mock agent for testing"

    def _register_skills(self):
        """Register skills - empty for mock."""
        pass

    def get_tools(self):
        return []

    def can_handle(self, intent: str) -> float:
        if "mock" in intent.lower():
            return 0.9
        return 0.0


class TestRoutingRules:
    """Tests for routing rules."""

    def test_routing_rules_exist(self):
        """Test that routing rules are defined."""
        assert len(ROUTING_RULES) > 0
        assert "flux_analysis" in ROUTING_RULES
        assert "scenario" in ROUTING_RULES
        assert "gene_analysis" in ROUTING_RULES
        assert "data_query" in ROUTING_RULES
        assert "strain_knowledge" in ROUTING_RULES

    def test_flux_analysis_keywords(self):
        """Test flux analysis keywords."""
        keywords = ROUTING_RULES["flux_analysis"]
        assert "fba" in keywords
        assert "fva" in keywords
        assert "플럭스" in keywords

    def test_scenario_keywords(self):
        """Test scenario keywords."""
        keywords = ROUTING_RULES["scenario"]
        assert "aerobic" in keywords
        assert "anaerobic" in keywords
        assert "혐기" in keywords


class TestAgentRegistry:
    """Tests for AgentRegistry class."""

    def test_registry_creation(self, agent_context):
        """Test creating a registry."""
        registry = AgentRegistry(agent_context)
        assert registry.context is agent_context
        assert len(registry.agent_types) == 0

    def test_register_agent_class(self, agent_context):
        """Test registering an agent class."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("mock", MockAgent)

        assert "mock" in registry.agent_types

    def test_register_agent_instance(self, agent_context):
        """Test registering an agent instance."""
        registry = AgentRegistry(agent_context)
        agent = MockAgent(agent_context)
        registry.register_agent("mock", agent)

        retrieved = registry.get_agent("mock")
        assert retrieved is agent

    def test_get_nonexistent_agent(self, agent_context):
        """Test getting nonexistent agent."""
        registry = AgentRegistry(agent_context)
        agent = registry.get_agent("nonexistent")

        assert agent is None

    def test_lazy_instantiation(self, agent_context):
        """Test lazy instantiation of agent class."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("mock", MockAgent)

        # Agent shouldn't be instantiated yet
        assert "mock" not in registry._agents

        # Get agent should instantiate it
        agent = registry.get_agent("mock")
        assert agent is not None
        assert "mock" in registry._agents

    def test_get_all_agents(self, agent_context):
        """Test getting all agents."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("mock1", MockAgent)
        registry.register_agent_class("mock2", MockAgent)

        agents = registry.get_all_agents()
        assert len(agents) == 2
        assert "mock1" in agents
        assert "mock2" in agents


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_english(self, agent_context):
        """Test detecting English text."""
        registry = AgentRegistry(agent_context)
        lang = registry.detect_language("This is English text")

        assert lang == "en"

    def test_detect_korean(self, agent_context):
        """Test detecting Korean text."""
        registry = AgentRegistry(agent_context)
        lang = registry.detect_language("이것은 한국어 텍스트입니다")

        assert lang == "ko"

    def test_detect_mixed_mostly_korean(self, agent_context):
        """Test detecting mixed text with mostly Korean."""
        registry = AgentRegistry(agent_context)
        lang = registry.detect_language("FBA 분석을 수행해주세요")

        assert lang == "ko"

    def test_detect_mixed_mostly_english(self, agent_context):
        """Test detecting mixed text with mostly English."""
        registry = AgentRegistry(agent_context)
        lang = registry.detect_language("perform FBA analysis")

        assert lang == "en"


class TestIntentRouting:
    """Tests for intent-based routing."""

    def test_route_flux_intent(self, agent_context):
        """Test routing flux-related intent."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("flux_analysis", MockAgent)

        routes = registry.route_intent("perform FBA analysis")

        assert len(routes) > 0
        # flux_analysis should have score > 0
        agent_types = [r[0] for r in routes]
        assert "flux_analysis" in agent_types

    def test_route_scenario_intent(self, agent_context):
        """Test routing scenario-related intent."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("scenario", MockAgent)

        routes = registry.route_intent("set anaerobic condition")

        assert len(routes) > 0
        agent_types = [r[0] for r in routes]
        assert "scenario" in agent_types

    def test_route_korean_intent(self, agent_context):
        """Test routing Korean intent."""
        registry = AgentRegistry(agent_context)
        registry.register_agent_class("flux_analysis", MockAgent)

        routes = registry.route_intent("플럭스 분석해줘")

        assert len(routes) > 0

    def test_extract_keywords(self, agent_context):
        """Test keyword extraction."""
        registry = AgentRegistry(agent_context)
        keywords = registry.extract_keywords("perform FBA analysis with fva")

        assert "fba" in keywords
        assert "fva" in keywords


class TestDefaultRegistry:
    """Tests for default registry creation."""

    def test_create_default_registry(self, agent_context):
        """Test creating default registry."""
        registry = create_default_registry(agent_context)

        assert registry is not None
        assert len(registry.agent_types) > 0

    def test_default_registry_has_flux_analysis(self, agent_context):
        """Test that default registry has flux analysis agent."""
        registry = create_default_registry(agent_context)
        agent = registry.get_agent("flux_analysis")

        assert agent is not None
        assert agent.name == "flux_analysis"

    def test_default_registry_has_scenario(self, agent_context):
        """Test that default registry has scenario agent."""
        registry = create_default_registry(agent_context)
        agent = registry.get_agent("scenario")

        assert agent is not None
        assert agent.name == "scenario"

    def test_default_registry_has_gene_analysis(self, agent_context):
        """Test that default registry has gene analysis agent."""
        registry = create_default_registry(agent_context)
        agent = registry.get_agent("gene_analysis")

        assert agent is not None
        assert agent.name == "gene_analysis"

    def test_default_registry_has_data_query(self, agent_context):
        """Test that default registry has data query agent."""
        registry = create_default_registry(agent_context)
        agent = registry.get_agent("data_query")

        assert agent is not None
        assert agent.name == "data_query"
