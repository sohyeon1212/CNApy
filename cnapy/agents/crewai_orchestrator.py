"""CrewAI-based Orchestrator for CNApy Multi-Agent System

This module provides LLM-powered intent understanding and tool selection
using CrewAI framework. It wraps existing CNApy agents and skills as
CrewAI tools for intelligent natural language processing.

Features:
- LLM-based intent understanding (vs regex pattern matching)
- Automatic parameter extraction from natural language
- Complex multi-step workflow execution
- Cancellable long-running operations
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from cnapy.agents.base_agent import AgentContext, AgentResponse, SkillResult, SkillStatus

if TYPE_CHECKING:
    from cnapy.gui_elements.llm_analysis_dialog import LLMConfig

logger = logging.getLogger(__name__)


class CrewAIBackend(Enum):
    """Supported LLM backends for CrewAI."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ToolResult:
    """Result from a CrewAI tool execution."""

    success: bool
    message: str
    message_ko: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class CNApyCrewOrchestrator:
    """CrewAI-based orchestrator for CNApy natural language understanding.

    This orchestrator uses CrewAI and LLMs to understand complex natural
    language requests and route them to appropriate CNApy tools. It
    supports:

    1. Simple requests: Direct mapping to single skill execution
    2. Parametric analysis: "Growth rate 0~100% 10%씩 pFBA"
    3. Multi-step workflows: "Set anaerobic, knockout PGI, then FBA"
    4. Cancellable operations: Long-running analyses can be interrupted
    """

    def __init__(
        self,
        context: AgentContext,
        llm_config: "LLMConfig | None" = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        """Initialize the CrewAI orchestrator.

        Args:
            context: Shared agent context with model and appdata
            llm_config: LLM configuration for API access
            cancel_check: Callback to check if cancellation was requested
        """
        self.context = context
        self.llm_config = llm_config
        self.cancel_check = cancel_check or (lambda: False)
        self._crewai_available = False
        self._agents_initialized = False

        # Lazy-load CrewAI components
        self._crew = None
        self._tools = {}
        self._crewai_agents = {}

        # Import existing CNApy agents
        self._cnapy_agents = {}
        self._setup_cnapy_agents()

    def _setup_cnapy_agents(self):
        """Initialize existing CNApy agents."""
        from cnapy.agents.data_query_agent import DataQueryAgent
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent
        from cnapy.agents.gene_analysis_agent import GeneAnalysisAgent
        from cnapy.agents.scenario_manager_agent import ScenarioManagerAgent

        self._cnapy_agents = {
            "flux_analysis": FluxAnalysisAgent(self.context),
            "gene_analysis": GeneAnalysisAgent(self.context),
            "scenario": ScenarioManagerAgent(self.context),
            "data_query": DataQueryAgent(self.context),
        }

        # Try to import strain knowledge agent (requires LLM)
        try:
            from cnapy.agents.strain_knowledge_agent import StrainKnowledgeAgent

            self._cnapy_agents["strain_knowledge"] = StrainKnowledgeAgent(self.context, self.llm_config)
        except ImportError:
            logger.debug("StrainKnowledgeAgent not available")

    def _check_crewai_available(self) -> bool:
        """Check if CrewAI is available."""
        if self._crewai_available:
            return True

        try:
            import crewai  # noqa: F401

            self._crewai_available = True
            return True
        except ImportError:
            logger.warning("CrewAI not installed. Install with: pip install 'cnapy[ai-agents]'")
            return False

    def _check_llm_configured(self) -> bool:
        """Check if LLM is properly configured."""
        if not self.llm_config:
            return False

        # Check if API key is available
        if self.llm_config.provider == "anthropic" and not self.llm_config.anthropic_api_key:
            return False
        if self.llm_config.provider == "openai" and not self.llm_config.openai_api_key:
            return False
        if self.llm_config.provider == "google" and not self.llm_config.google_api_key:
            return False

        return True

    def _initialize_crewai(self):
        """Initialize CrewAI agents and tools."""
        if self._agents_initialized:
            return

        if not self._check_crewai_available():
            return

        if not self._check_llm_configured():
            logger.warning("LLM not configured. CrewAI features will be limited.")
            return

        try:
            self._setup_crewai_tools()
            self._setup_crewai_agents()
            self._agents_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize CrewAI: {e}")

    def _setup_crewai_tools(self):
        """Create CrewAI tools from existing CNApy skills."""
        try:
            from crewai.tools import tool
        except ImportError:
            logger.warning("crewai.tools not available")
            return

        # Create tools for flux analysis
        @tool("Perform FBA")
        def perform_fba() -> str:
            """Run Flux Balance Analysis to find optimal flux distribution."""
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("flux_analysis")
            if agent:
                result = agent.execute_skill("perform_fba", {})
                return result.get_message(self.context.current_language)
            return "Flux analysis agent not available."

        @tool("Perform pFBA")
        def perform_pfba() -> str:
            """Run Parsimonious FBA for minimum total flux solution."""
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("flux_analysis")
            if agent:
                result = agent.execute_skill("perform_pfba", {})
                return result.get_message(self.context.current_language)
            return "Flux analysis agent not available."

        @tool("Perform FVA")
        def perform_fva(fraction_of_optimum: float = 0.0) -> str:
            """Run Flux Variability Analysis to determine flux ranges.

            Args:
                fraction_of_optimum: Fraction of optimal objective to maintain (0.0-1.0)
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("flux_analysis")
            if agent:
                result = agent.execute_skill("perform_fva", {"fraction_of_optimum": fraction_of_optimum})
                return result.get_message(self.context.current_language)
            return "Flux analysis agent not available."

        @tool("Parametric Analysis")
        def parametric_analysis(
            analysis_type: str,
            parameter: str,
            start_percent: float,
            end_percent: float,
            step_percent: float = 10.0,
        ) -> str:
            """Run analysis while varying a parameter across a range.

            Use this for requests like "pFBA from 0% to 100% growth rate in 10% steps".

            Args:
                analysis_type: Type of analysis (fba, pfba, fva)
                parameter: Parameter to vary (growth_rate, oxygen_uptake)
                start_percent: Starting value as percentage (0-100)
                end_percent: Ending value as percentage (0-100)
                step_percent: Step size as percentage (default 10)
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("flux_analysis")
            if agent:
                result = agent.execute_skill(
                    "parametric_analysis",
                    {
                        "analysis_type": analysis_type,
                        "parameter": parameter,
                        "start_percent": start_percent,
                        "end_percent": end_percent,
                        "step_percent": step_percent,
                    },
                )
                return result.get_message(self.context.current_language)
            return "Flux analysis agent not available."

        @tool("Apply Culture Condition")
        def apply_condition(condition_name: str) -> str:
            """Apply a predefined culture condition.

            Args:
                condition_name: Condition name (aerobic, anaerobic, microaerobic)
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("scenario")
            if agent:
                result = agent.execute_skill("apply_condition", {"condition_name": condition_name})
                return result.get_message(self.context.current_language)
            return "Scenario manager agent not available."

        @tool("Knockout Gene")
        def knockout_gene(gene_id: str) -> str:
            """Simulate a single gene knockout and analyze the effect.

            Args:
                gene_id: Gene ID to knockout
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("gene_analysis")
            if agent:
                result = agent.execute_skill("knockout_gene", {"gene_id": gene_id})
                return result.get_message(self.context.current_language)
            return "Gene analysis agent not available."

        @tool("Find Essential Genes")
        def find_essential_genes(threshold: float = 0.01) -> str:
            """Find genes whose knockout leads to zero or near-zero growth.

            Args:
                threshold: Growth rate threshold (default 0.01 = 1%)
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("gene_analysis")
            if agent:
                result = agent.execute_skill("find_essential_genes", {"threshold": threshold})
                return result.get_message(self.context.current_language)
            return "Gene analysis agent not available."

        @tool("Get Model Info")
        def get_model_info() -> str:
            """Get information about the loaded metabolic model."""
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("data_query")
            if agent:
                result = agent.execute_skill("get_model_info", {})
                return result.get_message(self.context.current_language)
            return "Data query agent not available."

        @tool("Search Reactions")
        def search_reactions(query: str) -> str:
            """Search for reactions by name or ID.

            Args:
                query: Search query string
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("data_query")
            if agent:
                result = agent.execute_skill("search_reactions", {"query": query})
                return result.get_message(self.context.current_language)
            return "Data query agent not available."

        @tool("Set Carbon Source")
        def set_carbon_source(carbon_source: str, uptake_rate: float = -10.0) -> str:
            """Set the primary carbon source.

            Args:
                carbon_source: Carbon source name (glucose, xylose, glycerol, etc.)
                uptake_rate: Uptake rate (negative for consumption)
            """
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("scenario")
            if agent:
                result = agent.execute_skill(
                    "set_carbon_source",
                    {"carbon_source": carbon_source, "uptake_rate": uptake_rate},
                )
                return result.get_message(self.context.current_language)
            return "Scenario manager agent not available."

        @tool("Clear Scenario")
        def clear_scenario() -> str:
            """Clear all scenario flux values and reset to default."""
            if self.cancel_check():
                return "Operation cancelled."
            agent = self._cnapy_agents.get("scenario")
            if agent:
                result = agent.execute_skill("clear_scenario", {})
                return result.get_message(self.context.current_language)
            return "Scenario manager agent not available."

        # Store tools
        self._tools = {
            "perform_fba": perform_fba,
            "perform_pfba": perform_pfba,
            "perform_fva": perform_fva,
            "parametric_analysis": parametric_analysis,
            "apply_condition": apply_condition,
            "knockout_gene": knockout_gene,
            "find_essential_genes": find_essential_genes,
            "get_model_info": get_model_info,
            "search_reactions": search_reactions,
            "set_carbon_source": set_carbon_source,
            "clear_scenario": clear_scenario,
        }

    def _get_llm(self):
        """Get the LLM instance based on configuration."""
        if not self.llm_config:
            return None

        try:
            from crewai import LLM

            if self.llm_config.provider == "anthropic":
                return LLM(
                    model=f"anthropic/{self.llm_config.anthropic_model}",
                    api_key=self.llm_config.anthropic_api_key,
                )
            elif self.llm_config.provider == "openai":
                return LLM(
                    model=f"openai/{self.llm_config.openai_model}",
                    api_key=self.llm_config.openai_api_key,
                )
            elif self.llm_config.provider == "google":
                return LLM(
                    model=f"gemini/{self.llm_config.google_model}",
                    api_key=self.llm_config.google_api_key,
                )
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")

        return None

    def _setup_crewai_agents(self):
        """Create CrewAI agents with specialized roles."""
        try:
            from crewai import Agent
        except ImportError:
            logger.warning("crewai.Agent not available")
            return

        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM not available for CrewAI agents")
            return

        # Flux Analysis Expert
        self._crewai_agents["flux_analyst"] = Agent(
            role="Flux Balance Analysis Expert",
            goal="Perform and interpret flux-based metabolic analyses",
            backstory=(
                "You are an expert in constraint-based modeling with deep knowledge "
                "of FBA, pFBA, FVA, and flux sampling. You understand metabolic "
                "network analysis and can interpret flux distributions."
            ),
            tools=[
                self._tools["perform_fba"],
                self._tools["perform_pfba"],
                self._tools["perform_fva"],
                self._tools["parametric_analysis"],
            ],
            llm=llm,
            verbose=False,
        )

        # Gene Analysis Specialist
        self._crewai_agents["gene_analyst"] = Agent(
            role="Gene Analysis Specialist",
            goal="Analyze gene essentiality and knockout effects",
            backstory=(
                "You are a specialist in gene-protein-reaction (GPR) rules, "
                "gene knockouts, and synthetic lethality analysis. You can "
                "identify essential genes and predict knockout effects."
            ),
            tools=[
                self._tools["knockout_gene"],
                self._tools["find_essential_genes"],
            ],
            llm=llm,
            verbose=False,
        )

        # Scenario Manager
        self._crewai_agents["scenario_manager"] = Agent(
            role="Culture Condition Manager",
            goal="Configure metabolic scenarios and culture conditions",
            backstory=(
                "You are an expert in setting up metabolic scenarios including "
                "aerobic/anaerobic conditions, carbon sources, and growth "
                "constraints. You understand microbial physiology."
            ),
            tools=[
                self._tools["apply_condition"],
                self._tools["set_carbon_source"],
                self._tools["clear_scenario"],
            ],
            llm=llm,
            verbose=False,
        )

        # Data Query Specialist
        self._crewai_agents["data_query"] = Agent(
            role="Model Information Specialist",
            goal="Query and retrieve metabolic model information",
            backstory=(
                "You are an expert in navigating metabolic models, searching "
                "for reactions, metabolites, genes, and pathways. You can "
                "quickly find relevant information."
            ),
            tools=[
                self._tools["get_model_info"],
                self._tools["search_reactions"],
            ],
            llm=llm,
            verbose=False,
        )

    def route(
        self,
        user_message: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> AgentResponse:
        """Route a user message using CrewAI for intent understanding.

        This method first tries to use CrewAI for LLM-based intent
        understanding. If CrewAI is not available or fails, it falls
        back to the traditional regex-based routing.

        Args:
            user_message: Natural language user message
            cancel_check: Optional callback to check for cancellation

        Returns:
            AgentResponse with results from execution
        """
        if cancel_check:
            self.cancel_check = cancel_check

        # Check if we can use CrewAI
        if self._check_crewai_available() and self._check_llm_configured():
            try:
                self._initialize_crewai()
                return self._route_with_crewai(user_message)
            except Exception as e:
                logger.warning(f"CrewAI routing failed, falling back: {e}")

        # Fall back to traditional routing
        return self._fallback_route(user_message)

    def _route_with_crewai(self, user_message: str) -> AgentResponse:
        """Route using CrewAI for intent understanding.

        Args:
            user_message: Natural language user message

        Returns:
            AgentResponse with results
        """
        try:
            from crewai import Crew, Process, Task
        except ImportError:
            return self._fallback_route(user_message)

        if not self._crewai_agents:
            return self._fallback_route(user_message)

        # Detect language
        language = "ko" if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in user_message) else "en"
        self.context.current_language = language

        # Create a task for intent analysis and execution
        analysis_task = Task(
            description=f"""
            Analyze the user's request and perform the appropriate metabolic analysis.

            User request: {user_message}

            Instructions:
            1. Understand what analysis the user wants
            2. Extract any parameters (conditions, values, ranges)
            3. If the user asks for parametric analysis (e.g., "pFBA from 0 to 100% growth"),
               use the parametric_analysis tool
            4. Execute the appropriate tool(s)
            5. Return results in a clear format

            Common request patterns:
            - "FBA" or "FBA 수행" -> perform_fba
            - "pFBA" or "파시모니어스" -> perform_pfba
            - "FVA" or "플럭스 변동성" -> perform_fva
            - "Growth rate 0~100% 10%씩 pFBA" -> parametric_analysis with
              analysis_type="pfba", parameter="growth_rate",
              start_percent=0, end_percent=100, step_percent=10
            - "aerobic" or "호기" -> apply_condition("aerobic")
            - "anaerobic" or "혐기" -> apply_condition("anaerobic")
            - "knockout [gene]" -> knockout_gene
            - "essential genes" -> find_essential_genes
            """,
            expected_output="Analysis results with interpretation",
            agent=self._select_best_agent(user_message),
        )

        # Create and run crew
        crew = Crew(
            agents=list(self._crewai_agents.values()),
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            result_str = str(result)

            return AgentResponse(
                success=True,
                message=result_str,
                message_ko=result_str if language == "ko" else "",
                agent_name="crewai",
                metadata={"routing": "crewai", "language": language},
            )
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                message_ko=f"분석 실패: {str(e)}",
                agent_name="crewai",
                metadata={"error": str(e)},
            )

    def _select_best_agent(self, message: str):
        """Select the best CrewAI agent for the given message.

        Args:
            message: User message

        Returns:
            Best matching CrewAI agent
        """
        message_lower = message.lower()

        # Keyword-based agent selection
        flux_keywords = ["fba", "pfba", "fva", "flux", "플럭스", "최적화", "변동성", "sampling"]
        gene_keywords = ["gene", "knockout", "ko", "essential", "유전자", "녹아웃", "필수"]
        scenario_keywords = ["aerobic", "anaerobic", "condition", "carbon", "호기", "혐기", "조건", "탄소"]
        query_keywords = ["info", "search", "find", "정보", "검색", "찾"]

        for keyword in flux_keywords:
            if keyword in message_lower:
                return self._crewai_agents.get("flux_analyst")

        for keyword in gene_keywords:
            if keyword in message_lower:
                return self._crewai_agents.get("gene_analyst")

        for keyword in scenario_keywords:
            if keyword in message_lower:
                return self._crewai_agents.get("scenario_manager")

        for keyword in query_keywords:
            if keyword in message_lower:
                return self._crewai_agents.get("data_query")

        # Default to flux analyst
        return self._crewai_agents.get("flux_analyst")

    def _fallback_route(self, user_message: str) -> AgentResponse:
        """Fall back to traditional orchestrator routing.

        Args:
            user_message: Natural language user message

        Returns:
            AgentResponse from traditional routing
        """
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orchestrator = OrchestratorAgent(self.context, self.llm_config)
        return orchestrator.route(user_message)

    def execute_tool_directly(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> SkillResult:
        """Execute a CNApy tool directly without CrewAI.

        This is useful for quick actions and button-triggered operations.

        Args:
            tool_name: Name of the tool/skill to execute
            params: Parameters for the tool

        Returns:
            SkillResult from execution
        """
        # Map tool names to agents and skills
        tool_to_agent = {
            "perform_fba": ("flux_analysis", "perform_fba"),
            "perform_pfba": ("flux_analysis", "perform_pfba"),
            "perform_fva": ("flux_analysis", "perform_fva"),
            "parametric_analysis": ("flux_analysis", "parametric_analysis"),
            "apply_condition": ("scenario", "apply_condition"),
            "knockout_gene": ("gene_analysis", "knockout_gene"),
            "find_essential_genes": ("gene_analysis", "find_essential_genes"),
            "get_model_info": ("data_query", "get_model_info"),
            "search_reactions": ("data_query", "search_reactions"),
            "set_carbon_source": ("scenario", "set_carbon_source"),
            "clear_scenario": ("scenario", "clear_scenario"),
        }

        if tool_name not in tool_to_agent:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Unknown tool: {tool_name}",
                message=f"Tool '{tool_name}' not found.",
                message_ko=f"'{tool_name}' 도구를 찾을 수 없습니다.",
            )

        agent_name, skill_name = tool_to_agent[tool_name]
        agent = self._cnapy_agents.get(agent_name)

        if agent is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Agent not available: {agent_name}",
                message=f"Agent '{agent_name}' is not available.",
                message_ko=f"'{agent_name}' 에이전트를 사용할 수 없습니다.",
            )

        return agent.execute_skill(skill_name, params)

    def get_available_tools(self) -> list[dict[str, str]]:
        """Get list of available tools with descriptions.

        Returns:
            List of tool info dictionaries
        """
        tools = []

        for agent_name, agent in self._cnapy_agents.items():
            for skill_name, skill in agent.skills.items():
                tools.append(
                    {
                        "name": skill_name,
                        "agent": agent_name,
                        "description": skill.description,
                        "description_ko": skill.description_ko,
                    }
                )

        return tools

    def is_crewai_available(self) -> bool:
        """Check if CrewAI is available and configured.

        Returns:
            True if CrewAI can be used
        """
        return self._check_crewai_available() and self._check_llm_configured()
