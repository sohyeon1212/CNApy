"""Base Agent Module for CNApy Multi-Agent System

This module defines the core abstractions for the agent system:
- BaseAgent: Abstract base class for all sub-agents
- AgentContext: Shared context between agents (model, scenario, etc.)
- SkillResult: Standard result format from agent skill execution
- Skill: Definition of an agent capability
- ToolDefinition: LLM function calling tool definition
- AgentResponse: Standard response from agent routing
- WorkflowStep: Single step in a multi-step workflow
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import cobra
    from cnapy.appdata import AppData, Scenario


class SkillStatus(Enum):
    """Status of a skill execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class SkillResult:
    """Result from executing an agent skill.

    Attributes:
        status: Execution status (success, failure, partial, pending)
        data: Result data (varies by skill)
        message: Human-readable message describing the result
        message_ko: Korean version of the message (optional)
        error: Error message if status is failure
        metadata: Additional metadata about the execution
    """
    status: SkillStatus
    data: Any = None
    message: str = ""
    message_ko: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the skill execution was successful."""
        return self.status == SkillStatus.SUCCESS

    def get_message(self, language: str = "en") -> str:
        """Get message in the specified language.

        Args:
            language: Language code ("en" or "ko")

        Returns:
            Message in the specified language, falls back to English if Korean not available
        """
        if language == "ko" and self.message_ko:
            return self.message_ko
        return self.message


@dataclass
class Skill:
    """Definition of an agent skill/capability.

    Attributes:
        name: Unique skill identifier
        description: Human-readable description
        description_ko: Korean description
        parameters: Parameter definitions for the skill
        required_params: List of required parameter names
        handler: Function that executes the skill
    """
    name: str
    description: str
    description_ko: str = ""
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., SkillResult]] = None

    def to_tool_definition(self) -> "ToolDefinition":
        """Convert skill to LLM tool definition format."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
        )


@dataclass
class ToolDefinition:
    """LLM function calling tool definition.

    Follows the OpenAI/Anthropic function calling format.

    Attributes:
        name: Function/tool name
        description: Description of what the tool does
        parameters: JSON Schema for the parameters
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class WorkflowStep:
    """Single step in a multi-step workflow.

    Attributes:
        agent_name: Name of the agent to execute
        skill_name: Name of the skill to execute
        params: Parameters for the skill
        condition: Optional condition to check before execution
        on_success: Next step name on success (or None to continue)
        on_failure: Next step name on failure (or None to stop)
    """
    agent_name: str
    skill_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None


@dataclass
class AgentResponse:
    """Standard response from agent routing/execution.

    Attributes:
        success: Whether the overall operation succeeded
        message: Human-readable response message
        message_ko: Korean version of the message
        results: List of skill results from execution
        data: Aggregated data from all results
        agent_name: Name of the agent(s) that handled the request
        metadata: Additional response metadata
    """
    success: bool
    message: str
    message_ko: str = ""
    results: List[SkillResult] = field(default_factory=list)
    data: Any = None
    agent_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_message(self, language: str = "en") -> str:
        """Get message in the specified language."""
        if language == "ko" and self.message_ko:
            return self.message_ko
        return self.message


@dataclass
class AgentContext:
    """Shared context between agents.

    Contains references to the current model, scenario, application data,
    and other shared state needed by all agents.

    Attributes:
        appdata: CNApy AppData instance
        main_window: Reference to the main window (optional, for GUI operations)
        analysis_results: Cache of analysis results
        conversation_history: List of conversation messages
        current_language: Current UI language ("en" or "ko")
    """
    appdata: "AppData"
    main_window: Optional[Any] = None
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_language: str = "en"

    @property
    def model(self) -> Optional["cobra.Model"]:
        """Get the current COBRA model."""
        if self.appdata and self.appdata.project:
            return self.appdata.project.cobra_py_model
        return None

    @property
    def scenario(self) -> Optional["Scenario"]:
        """Get the current scenario."""
        if self.appdata and self.appdata.project:
            return self.appdata.project.scen_values
        return None

    @property
    def comp_values(self) -> Dict[str, Tuple[float, float]]:
        """Get the current computed values (flux values)."""
        if self.appdata and self.appdata.project:
            return self.appdata.project.comp_values
        return {}

    @property
    def solution(self) -> Optional[Any]:
        """Get the current COBRA solution."""
        if self.appdata and self.appdata.project:
            return self.appdata.project.solution
        return None

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
        })

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Get the most recent n messages from conversation history.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of recent messages
        """
        return self.conversation_history[-n:] if self.conversation_history else []

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()

    def cache_result(self, key: str, result: Any):
        """Cache an analysis result.

        Args:
            key: Cache key
            result: Result to cache
        """
        self.analysis_results[key] = result

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached analysis result.

        Args:
            key: Cache key

        Returns:
            Cached result or None if not found
        """
        return self.analysis_results.get(key)


class BaseAgent(ABC):
    """Abstract base class for all sub-agents.

    Each agent handles a specific domain of functionality:
    - FluxAnalysisAgent: FBA, pFBA, FVA, MOMA, ROOM, sampling
    - GeneAnalysisAgent: Gene knockouts, essential genes
    - ScenarioManagerAgent: Condition settings, scenario save/load
    - DataQueryAgent: Model information queries
    - StrainKnowledgeAgent: LLM-based strain knowledge

    Attributes:
        name: Agent name/identifier
        description: Agent description
        description_ko: Korean description
        context: Shared agent context
        skills: Dictionary of available skills
    """

    def __init__(self, context: AgentContext):
        """Initialize the agent.

        Args:
            context: Shared agent context
        """
        self.context = context
        self._skills: Dict[str, Skill] = {}
        self._register_skills()

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name/identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description in English."""
        pass

    @property
    def description_ko(self) -> str:
        """Agent description in Korean. Override in subclasses."""
        return self.description

    @property
    def skills(self) -> Dict[str, Skill]:
        """Dictionary of available skills."""
        return self._skills

    @abstractmethod
    def _register_skills(self):
        """Register all skills for this agent. Override in subclasses."""
        pass

    def register_skill(self, skill: Skill):
        """Register a skill.

        Args:
            skill: Skill to register
        """
        self._skills[skill.name] = skill

    def get_tools(self) -> List[ToolDefinition]:
        """Get LLM function calling tool definitions.

        Returns:
            List of tool definitions for all skills
        """
        return [skill.to_tool_definition() for skill in self._skills.values()]

    def execute_skill(self, skill_name: str, params: Dict[str, Any]) -> SkillResult:
        """Execute a skill by name.

        Args:
            skill_name: Name of the skill to execute
            params: Parameters for the skill

        Returns:
            SkillResult from execution
        """
        if skill_name not in self._skills:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Unknown skill: {skill_name}",
                message=f"Skill '{skill_name}' not found in {self.name}",
                message_ko=f"'{skill_name}' 스킬을 {self.name}에서 찾을 수 없습니다.",
            )

        skill = self._skills[skill_name]

        # Check required parameters
        for param in skill.required_params:
            if param not in params:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error=f"Missing required parameter: {param}",
                    message=f"Missing required parameter '{param}' for skill '{skill_name}'",
                    message_ko=f"'{skill_name}' 스킬에 필수 파라미터 '{param}'가 누락되었습니다.",
                )

        # Execute the skill handler
        if skill.handler:
            try:
                return skill.handler(**params)
            except Exception as e:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error=str(e),
                    message=f"Error executing skill '{skill_name}': {str(e)}",
                    message_ko=f"'{skill_name}' 스킬 실행 중 오류 발생: {str(e)}",
                )
        else:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No handler defined",
                message=f"No handler defined for skill '{skill_name}'",
                message_ko=f"'{skill_name}' 스킬에 핸들러가 정의되지 않았습니다.",
            )

    def can_handle(self, intent: str, keywords: List[str] = None) -> float:
        """Calculate probability that this agent can handle the given intent.

        Args:
            intent: User intent description or request
            keywords: Optional list of extracted keywords

        Returns:
            Probability score from 0.0 to 1.0
        """
        # Default implementation: check against skill names and descriptions
        intent_lower = intent.lower()
        keywords_lower = [k.lower() for k in (keywords or [])]

        score = 0.0
        max_score = 0.0

        for skill in self._skills.values():
            # Check skill name
            if skill.name.lower() in intent_lower:
                score += 0.5

            # Check skill description
            desc_words = skill.description.lower().split()
            for word in desc_words:
                if len(word) > 3 and word in intent_lower:
                    score += 0.1

            # Check keywords
            for keyword in keywords_lower:
                if keyword in skill.name.lower() or keyword in skill.description.lower():
                    score += 0.2

            max_score += 1.0

        return min(1.0, score / max(max_score, 1.0))

    def get_description(self, language: str = "en") -> str:
        """Get agent description in the specified language.

        Args:
            language: Language code ("en" or "ko")

        Returns:
            Agent description
        """
        if language == "ko":
            return self.description_ko
        return self.description
