"""Agent Registry for CNApy Multi-Agent System

This module provides centralized registration and management of all agents.
It handles:
- Agent registration and lookup
- Intent-to-agent routing based on keywords
- Agent lifecycle management
"""

from typing import Dict, List, Optional, Type, Tuple
import re

from cnapy.agents.base_agent import BaseAgent, AgentContext


# Routing rules for intent detection
# Maps keywords to agent types
ROUTING_RULES: Dict[str, List[str]] = {
    "flux_analysis": [
        # English keywords
        "fba", "pfba", "fva", "moma", "room", "sampling", "flux",
        "optimize", "optimization", "balance", "variability",
        # Korean keywords
        "플럭스", "최적화", "분석", "샘플링", "변동성",
    ],
    "gene_analysis": [
        # English keywords
        "knockout", "ko", "essential", "gene", "deletion", "lethal",
        "synthetic", "double", "single",
        # Korean keywords
        "녹아웃", "유전자", "필수", "치사", "삭제",
    ],
    "scenario": [
        # English keywords
        "condition", "scenario", "save", "load", "setting", "bound",
        "carbon", "nitrogen", "oxygen", "aerobic", "anaerobic",
        "source", "objective", "constraint",
        # Korean keywords
        "조건", "시나리오", "저장", "로드", "설정", "경계",
        "탄소원", "질소원", "산소", "호기", "혐기", "목적함수",
    ],
    "data_query": [
        # English keywords
        "info", "information", "search", "find", "query", "list",
        "reaction", "metabolite", "pathway", "what", "which", "show",
        "get", "display",
        # Korean keywords
        "정보", "검색", "찾기", "조회", "목록", "반응", "대사체",
        "경로", "무엇", "어떤", "보여", "표시",
    ],
    "strain_knowledge": [
        # English keywords
        "strain", "organism", "species", "exist", "presence", "have",
        "literature", "paper", "research", "compare", "suggest",
        # Korean keywords
        "균주", "생물", "종", "존재", "있", "문헌", "논문",
        "연구", "비교", "제안",
    ],
}


class AgentRegistry:
    """Central registry for all agents.

    Handles agent registration, lookup, and intent-based routing.

    Attributes:
        context: Shared agent context
        agents: Dictionary of registered agents
    """

    def __init__(self, context: AgentContext):
        """Initialize the registry.

        Args:
            context: Shared agent context
        """
        self.context = context
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}

    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]):
        """Register an agent class for later instantiation.

        Args:
            agent_type: Type identifier (e.g., "flux_analysis")
            agent_class: Agent class to register
        """
        self._agent_classes[agent_type] = agent_class

    def register_agent(self, agent_type: str, agent: BaseAgent):
        """Register an agent instance.

        Args:
            agent_type: Type identifier (e.g., "flux_analysis")
            agent: Agent instance to register
        """
        self._agents[agent_type] = agent

    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get an agent by type.

        Args:
            agent_type: Type identifier

        Returns:
            Agent instance or None if not found
        """
        # Return existing instance if available
        if agent_type in self._agents:
            return self._agents[agent_type]

        # Create new instance if class is registered
        if agent_type in self._agent_classes:
            agent = self._agent_classes[agent_type](self.context)
            self._agents[agent_type] = agent
            return agent

        return None

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents.

        Returns:
            Dictionary of agent type to agent instance
        """
        # Ensure all registered classes are instantiated
        for agent_type in self._agent_classes:
            if agent_type not in self._agents:
                self._agents[agent_type] = self._agent_classes[agent_type](self.context)
        return self._agents

    def route_intent(self, user_message: str) -> List[Tuple[str, float]]:
        """Route a user message to appropriate agent(s).

        Uses keyword matching and agent can_handle scoring.

        Args:
            user_message: User's natural language message

        Returns:
            List of (agent_type, score) tuples, sorted by score descending
        """
        message_lower = user_message.lower()
        scores: Dict[str, float] = {}

        # Step 1: Keyword-based routing
        for agent_type, keywords in ROUTING_RULES.items():
            keyword_score = 0.0
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    keyword_score += 0.3
            scores[agent_type] = min(keyword_score, 1.0)

        # Step 2: Agent-based scoring (if agents are instantiated)
        for agent_type, agent in self._agents.items():
            agent_score = agent.can_handle(user_message)
            # Combine scores (weighted average)
            current = scores.get(agent_type, 0.0)
            scores[agent_type] = current * 0.6 + agent_score * 0.4

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter out zero scores
        return [(agent_type, score) for agent_type, score in sorted_scores if score > 0.0]

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text.

        Uses simple heuristics based on character ranges.

        Args:
            text: Input text

        Returns:
            Language code ("ko" for Korean, "en" for English)
        """
        # Count Korean characters (Hangul)
        korean_pattern = re.compile(r'[\uAC00-\uD7A3]')
        korean_chars = len(korean_pattern.findall(text))

        # If more than 10% of characters are Korean, assume Korean
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0 and korean_chars / total_chars > 0.1:
            return "ko"

        return "en"

    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        # Collect all keywords from routing rules
        all_keywords = set()
        for keywords in ROUTING_RULES.values():
            all_keywords.update(k.lower() for k in keywords)

        # Find matching keywords in text
        text_lower = text.lower()
        found = []
        for keyword in all_keywords:
            if keyword in text_lower:
                found.append(keyword)

        return found

    @property
    def agent_types(self) -> List[str]:
        """List of all registered agent types."""
        all_types = set(self._agents.keys()) | set(self._agent_classes.keys())
        return list(all_types)


def create_default_registry(context: AgentContext) -> AgentRegistry:
    """Create a registry with all default agents registered.

    Args:
        context: Shared agent context

    Returns:
        Configured AgentRegistry instance
    """
    registry = AgentRegistry(context)

    # Import and register agent classes
    # These imports are deferred to avoid circular imports
    try:
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent
        registry.register_agent_class("flux_analysis", FluxAnalysisAgent)
    except ImportError:
        pass

    try:
        from cnapy.agents.gene_analysis_agent import GeneAnalysisAgent
        registry.register_agent_class("gene_analysis", GeneAnalysisAgent)
    except ImportError:
        pass

    try:
        from cnapy.agents.scenario_manager_agent import ScenarioManagerAgent
        registry.register_agent_class("scenario", ScenarioManagerAgent)
    except ImportError:
        pass

    try:
        from cnapy.agents.data_query_agent import DataQueryAgent
        registry.register_agent_class("data_query", DataQueryAgent)
    except ImportError:
        pass

    try:
        from cnapy.agents.strain_knowledge_agent import StrainKnowledgeAgent
        registry.register_agent_class("strain_knowledge", StrainKnowledgeAgent)
    except ImportError:
        pass

    return registry
