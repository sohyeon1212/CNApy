"""Orchestrator Agent for CNApy Multi-Agent System

This is the main routing agent that:
- Analyzes user intent from natural language requests
- Routes to appropriate sub-agents
- Manages multi-step workflows
- Integrates LLM for intent understanding (optional)
- Provides unified response formatting
"""

import re
from typing import Any

from cnapy.agents.agent_registry import create_default_registry
from cnapy.agents.base_agent import (
    AgentContext,
    AgentResponse,
    SkillResult,
    SkillStatus,
    WorkflowStep,
)
from cnapy.agents.predefined_scenarios import get_workflow


class OrchestratorAgent:
    """Main orchestrator that routes requests to appropriate sub-agents.

    The orchestrator:
    1. Analyzes user intent from natural language
    2. Determines which agent(s) should handle the request
    3. Extracts parameters from the request
    4. Executes single or multi-step workflows
    5. Aggregates and formats responses

    Attributes:
        context: Shared agent context
        registry: Agent registry for routing
        llm_config: LLM configuration (optional, for advanced intent understanding)
    """

    # Intent patterns for rule-based routing
    INTENT_PATTERNS = {
        # Flux Analysis patterns
        "perform_fba": [
            r"\bfba\b",
            r"flux\s*balance",
            r"최적화",
            r"optimize",
            r"성장\s*률",
            r"growth\s*rate",
            r"플럭스\s*분석",
        ],
        "perform_pfba": [
            r"\bpfba\b",
            r"parsimonious",
            r"최소\s*플럭스",
        ],
        "perform_fva": [
            r"\bfva\b",
            r"variability",
            r"변동성",
            r"플럭스\s*범위",
        ],
        "perform_moma": [
            r"\bmoma\b",
            r"minimization.*adjustment",
        ],
        "perform_room": [
            r"\broom\b",
            r"regulatory.*on.*off",
        ],
        "perform_flux_sampling": [
            r"sampling",
            r"샘플링",
            r"sample\s*flux",
        ],
        # Scenario Management patterns
        "apply_condition": [
            r"aerobic|anaerobic|microaerobic",
            r"호기|혐기|미호기",
            r"조건\s*설정",
            r"set.*condition",
            r"apply.*condition",
        ],
        "set_carbon_source": [
            r"탄소원",
            r"carbon\s*source",
            r"glucose|xylose|glycerol",
            r"포도당|자일로스|글리세롤",
        ],
        "set_nitrogen_source": [
            r"질소원",
            r"nitrogen\s*source",
            r"ammonium|ammonia",
            r"암모니아|암모늄",
        ],
        # Gene Analysis patterns
        "knockout_gene": [
            r"knockout|ko|녹아웃|삭제",
            r"gene.*delete",
        ],
        "find_essential_genes": [
            r"essential.*gene",
            r"필수.*유전자",
            r"lethal",
        ],
        # Data Query patterns
        "get_model_info": [
            r"model.*info",
            r"모델.*정보",
            r"how\s*many",
        ],
        "search_reactions": [
            r"search.*reaction",
            r"반응.*검색",
            r"find.*reaction",
        ],
        # Strain Knowledge patterns
        "analyze_strain": [
            r"strain.*exist",
            r"균주",
            r"있어\?|있나요|존재",
        ],
        "compare_strains": [
            r"compare.*strain",
            r"균주.*비교",
            r"strain.*vs|versus",
            r"차이점|difference",
            r"strain.*differ",
        ],
        "suggest_modifications": [
            r"suggest.*modif|engineer",
            r"제안|개량|엔지니어링",
            r"produce|생산|합성",
            r"strategy|전략",
        ],
        "literature_search": [
            r"literature|paper|research",
            r"문헌|논문|연구",
            r"published|publication",
            r"what.*know",
        ],
        "check_reaction_in_strain": [
            r"(reaction|반응).*(exist|있|존재).*(strain|균주)",
            r"(strain|균주).*(have|가지|있).*(reaction|반응)",
        ],
        "check_gene_in_strain": [
            r"(gene|유전자).*(exist|있|존재).*(strain|균주)",
            r"(strain|균주).*(have|가지|있).*(gene|유전자)",
            r"ortholog|오솔로그|homolog",
        ],
    }

    # Map skills to agents
    SKILL_TO_AGENT = {
        "perform_fba": "flux_analysis",
        "perform_pfba": "flux_analysis",
        "perform_fva": "flux_analysis",
        "perform_moma": "flux_analysis",
        "perform_room": "flux_analysis",
        "perform_flux_sampling": "flux_analysis",
        "analyze_flux_distribution": "flux_analysis",
        "get_objective_value": "flux_analysis",
        "compare_flux_states": "flux_analysis",
        "apply_condition": "scenario",
        "set_reaction_bounds": "scenario",
        "set_carbon_source": "scenario",
        "set_nitrogen_source": "scenario",
        "set_objective": "scenario",
        "get_current_scenario": "scenario",
        "clear_scenario": "scenario",
        "save_scenario": "scenario",
        "load_scenario": "scenario",
        "list_conditions": "scenario",
        "list_carbon_sources": "scenario",
        "list_nitrogen_sources": "scenario",
        "knockout_gene": "gene_analysis",
        "knockout_genes": "gene_analysis",
        "find_essential_genes": "gene_analysis",
        "find_essential_reactions": "gene_analysis",
        "single_gene_ko_scan": "gene_analysis",
        "get_model_info": "data_query",
        "get_reaction_info": "data_query",
        "get_metabolite_info": "data_query",
        "get_gene_info": "data_query",
        "search_reactions": "data_query",
        "search_metabolites": "data_query",
        "search_genes": "data_query",
        "analyze_strain_reactions": "strain_knowledge",
        "analyze_strain_genes": "strain_knowledge",
        "get_strain_metabolism": "strain_knowledge",
        "compare_strains": "strain_knowledge",
        "suggest_modifications": "strain_knowledge",
        "literature_search": "strain_knowledge",
        "check_reaction_in_strain": "strain_knowledge",
        "check_gene_in_strain": "strain_knowledge",
        "analyze_strain": "strain_knowledge",
    }

    def __init__(self, context: AgentContext, llm_config: Any | None = None):
        """Initialize the orchestrator.

        Args:
            context: Shared agent context
            llm_config: Optional LLM configuration for advanced intent understanding
        """
        self.context = context
        self.llm_config = llm_config
        self.registry = create_default_registry(context)

    def route(self, user_message: str) -> AgentResponse:
        """Route a user message to appropriate agent(s).

        This is the main entry point for processing user requests.

        Args:
            user_message: Natural language user message

        Returns:
            AgentResponse with results from execution
        """
        # Detect language
        language = self.registry.detect_language(user_message)
        self.context.current_language = language

        # Add to conversation history
        self.context.add_message("user", user_message)

        # Step 1: Analyze intent
        intent_analysis = self._analyze_intent(user_message)

        # Step 2: Check for predefined workflows
        workflow = self._check_predefined_workflow(user_message)
        if workflow:
            return self._execute_workflow(workflow, language)

        # Step 3: Execute based on intent analysis
        if intent_analysis["skill"]:
            return self._execute_single_skill(intent_analysis, language)

        # Step 4: Try LLM-based routing if available
        if self.llm_config:
            llm_response = self._route_with_llm(user_message)
            if llm_response:
                return llm_response

        # Step 5: Fallback - try to provide helpful response
        return self._generate_fallback_response(user_message, language)

    def _analyze_intent(self, message: str) -> dict[str, Any]:
        """Analyze user intent using rule-based patterns.

        Args:
            message: User message

        Returns:
            Dictionary with detected skill, agent, and parameters
        """
        message_lower = message.lower()

        # Try to match intent patterns
        best_match = None
        best_score = 0

        for skill_name, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            if score > best_score:
                best_score = score
                best_match = skill_name

        if best_match and best_score > 0:
            agent_type = self.SKILL_TO_AGENT.get(best_match)
            params = self._extract_parameters(message, best_match)
            return {
                "skill": best_match,
                "agent": agent_type,
                "params": params,
                "confidence": min(best_score / 3, 1.0),
            }

        # Try registry-based routing
        routed = self.registry.route_intent(message)
        if routed:
            top_agent, confidence = routed[0]
            return {
                "skill": None,
                "agent": top_agent,
                "params": {},
                "confidence": confidence,
            }

        return {
            "skill": None,
            "agent": None,
            "params": {},
            "confidence": 0.0,
        }

    def _extract_parameters(self, message: str, skill_name: str) -> dict[str, Any]:
        """Extract parameters from the message for a specific skill.

        Args:
            message: User message
            skill_name: Name of the skill to extract parameters for

        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        message_lower = message.lower()

        # FVA fraction extraction
        if skill_name == "perform_fva":
            fraction_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", message)
            if fraction_match:
                value = float(fraction_match.group(1))
                if value > 1:
                    value = value / 100  # Convert percentage
                params["fraction_of_optimum"] = value

        # Condition extraction
        if skill_name == "apply_condition":
            # Check more specific conditions first (anaerobic before aerobic)
            for condition in ["anaerobic", "microaerobic", "aerobic", "minimal_media", "rich_media"]:
                if condition in message_lower:
                    params["condition_name"] = condition
                    break
            # Korean detection
            if "호기" in message and "혐기" not in message and "미호기" not in message:
                params["condition_name"] = "aerobic"
            elif "혐기" in message:
                params["condition_name"] = "anaerobic"
            elif "미호기" in message:
                params["condition_name"] = "microaerobic"

        # Carbon source extraction
        if skill_name == "set_carbon_source":
            sources = ["glucose", "xylose", "glycerol", "acetate", "lactate", "succinate"]
            sources_ko = {"포도당": "glucose", "자일로스": "xylose", "글리세롤": "glycerol"}
            for source in sources:
                if source in message_lower:
                    params["carbon_source"] = source
                    break
            for ko, en in sources_ko.items():
                if ko in message:
                    params["carbon_source"] = en
                    break

        # Gene ID extraction
        if skill_name in ["knockout_gene", "knockout_genes"]:
            # Try to find gene IDs (common patterns)
            gene_patterns = [
                r"\b([a-z]{3}[A-Z])\b",  # E. coli gene pattern (e.g., pfkA)
                r"\b([A-Z]{3}\d+)\b",  # Yeast gene pattern (e.g., HXK1)
                r"'([^']+)'",  # Quoted identifiers
                r'"([^"]+)"',  # Double-quoted identifiers
            ]
            for pattern in gene_patterns:
                matches = re.findall(pattern, message)
                if matches:
                    if len(matches) == 1:
                        params["gene_id"] = matches[0]
                    else:
                        params["gene_ids"] = matches
                    break

        # Reaction ID extraction
        if skill_name in ["set_reaction_bounds", "set_objective", "get_reaction_info"]:
            # Common reaction patterns
            rxn_patterns = [
                r"\b(EX_\w+)\b",  # Exchange reactions
                r"\b(R_\w+)\b",  # Reactions with R_ prefix
                r"'([^']+)'",  # Quoted identifiers
                r'"([^"]+)"',  # Double-quoted identifiers
            ]
            for pattern in rxn_patterns:
                matches = re.findall(pattern, message)
                if matches:
                    params["reaction_id"] = matches[0]
                    break

        # Bounds extraction
        if skill_name == "set_reaction_bounds":
            bounds_match = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?", message)
            if bounds_match:
                params["lower_bound"] = float(bounds_match.group(1))
                params["upper_bound"] = float(bounds_match.group(2))

        # Essential genes threshold
        if skill_name == "find_essential_genes":
            threshold_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", message)
            if threshold_match:
                value = float(threshold_match.group(1))
                if value > 1:
                    value = value / 100
                params["threshold"] = value

        return params

    def _check_predefined_workflow(self, message: str) -> list[WorkflowStep] | None:
        """Check if the message matches a predefined workflow.

        Args:
            message: User message

        Returns:
            List of workflow steps or None
        """
        message_lower = message.lower()

        # Check for workflow keywords
        workflow_triggers = {
            "fba_aerobic": ["호기.*fba", "aerobic.*fba", "fba.*aerobic", "fba.*호기"],
            "fba_anaerobic": ["혐기.*fba", "anaerobic.*fba", "fba.*anaerobic", "fba.*혐기"],
            "essential_genes": ["필수.*유전자", "essential.*gene", "find.*essential"],
            "fva_analysis": [r"fva\s*분석", r"fva\s*analysis", r"flux\s*variability"],
        }

        for workflow_name, triggers in workflow_triggers.items():
            for trigger in triggers:
                if re.search(trigger, message_lower):
                    workflow = get_workflow(workflow_name)
                    if workflow:
                        return self._convert_workflow_to_steps(workflow)

        return None

    def _convert_workflow_to_steps(self, workflow) -> list[WorkflowStep]:
        """Convert a PredefinedWorkflow to WorkflowStep list.

        Args:
            workflow: PredefinedWorkflow instance

        Returns:
            List of WorkflowStep objects
        """
        steps = []
        for step_data in workflow.steps:
            steps.append(
                WorkflowStep(
                    agent_name=step_data["agent"],
                    skill_name=step_data["skill"],
                    params=step_data.get("params", {}),
                )
            )
        return steps

    def _execute_single_skill(self, intent: dict[str, Any], language: str) -> AgentResponse:
        """Execute a single skill based on intent analysis.

        Args:
            intent: Intent analysis result
            language: Language code

        Returns:
            AgentResponse
        """
        skill_name = intent["skill"]
        agent_type = intent["agent"]
        params = intent["params"]

        agent = self.registry.get_agent(agent_type)
        if agent is None:
            return AgentResponse(
                success=False,
                message=f"Agent '{agent_type}' is not available.",
                message_ko=f"'{agent_type}' 에이전트를 사용할 수 없습니다.",
                agent_name=agent_type,
            )

        # Execute the skill
        result = agent.execute_skill(skill_name, params)

        # Format response
        return AgentResponse(
            success=result.success,
            message=result.get_message(language),
            message_ko=result.message_ko,
            results=[result],
            data=result.data,
            agent_name=agent.name,
            metadata={
                "skill": skill_name,
                "params": params,
                "confidence": intent["confidence"],
            },
        )

    def _execute_workflow(self, steps: list[WorkflowStep], language: str) -> AgentResponse:
        """Execute a multi-step workflow.

        Args:
            steps: List of workflow steps
            language: Language code

        Returns:
            AgentResponse with aggregated results
        """
        results = []
        all_data = {}
        success = True

        for step in steps:
            agent = self.registry.get_agent(step.agent_name)
            if agent is None:
                results.append(
                    SkillResult(
                        status=SkillStatus.FAILURE,
                        error=f"Agent '{step.agent_name}' not available",
                        message=f"Agent '{step.agent_name}' is not available.",
                        message_ko=f"'{step.agent_name}' 에이전트를 사용할 수 없습니다.",
                    )
                )
                success = False
                break

            result = agent.execute_skill(step.skill_name, step.params)
            results.append(result)

            if result.data:
                all_data[f"{step.agent_name}_{step.skill_name}"] = result.data

            if not result.success:
                success = False
                break

        # Aggregate messages
        messages = [r.get_message(language) for r in results if r.message]
        messages_ko = [r.message_ko for r in results if r.message_ko]

        return AgentResponse(
            success=success,
            message=" → ".join(messages) if messages else "Workflow completed.",
            message_ko=" → ".join(messages_ko) if messages_ko else "워크플로우 완료.",
            results=results,
            data=all_data,
            metadata={"workflow": True, "steps": len(steps)},
        )

    def _route_with_llm(self, message: str) -> AgentResponse | None:
        """Use LLM to understand intent and route (if configured).

        Args:
            message: User message

        Returns:
            AgentResponse or None if LLM routing fails/unavailable
        """
        if not self.llm_config:
            return None

        # This is a placeholder for LLM-based routing
        # Will be implemented with the StrainKnowledgeAgent
        return None

    def _generate_fallback_response(self, message: str, language: str) -> AgentResponse:
        """Generate a helpful fallback response.

        Args:
            message: Original user message
            language: Language code

        Returns:
            AgentResponse with suggestions
        """
        # Get available capabilities
        agents = self.registry.get_all_agents()
        capabilities = []
        for _agent_type, agent in agents.items():
            skills = list(agent.skills.keys())[:3]  # Top 3 skills
            capabilities.append(f"{agent.name}: {', '.join(skills)}")

        if language == "ko":
            message = (
                "죄송합니다, 요청을 이해하지 못했습니다.\n\n"
                "다음과 같은 작업을 수행할 수 있습니다:\n"
                "- 'FBA 수행해줘' - 플럭스 균형 분석\n"
                "- '혐기 조건 설정해줘' - 배양 조건 변경\n"
                "- '필수 유전자 찾아줘' - 필수 유전자 분석\n"
                "- '모델 정보 알려줘' - 모델 정보 조회"
            )
            message_ko = message
        else:
            message = (
                "I'm sorry, I didn't understand the request.\n\n"
                "Here are some things I can do:\n"
                "- 'Perform FBA' - Flux Balance Analysis\n"
                "- 'Set anaerobic condition' - Change culture conditions\n"
                "- 'Find essential genes' - Essential gene analysis\n"
                "- 'Show model info' - Model information"
            )
            message_ko = ""

        return AgentResponse(
            success=False,
            message=message,
            message_ko=message_ko,
            metadata={"fallback": True},
        )

    def execute_workflow_by_name(self, workflow_name: str, params: dict[str, Any] = None) -> AgentResponse:
        """Execute a predefined workflow by name.

        Args:
            workflow_name: Name of the predefined workflow
            params: Optional parameters to inject into workflow steps

        Returns:
            AgentResponse
        """
        workflow = get_workflow(workflow_name)
        if workflow is None:
            return AgentResponse(
                success=False,
                message=f"Workflow '{workflow_name}' not found.",
                message_ko=f"'{workflow_name}' 워크플로우를 찾을 수 없습니다.",
            )

        steps = self._convert_workflow_to_steps(workflow)

        # Inject parameters if provided
        if params:
            for step in steps:
                for key, value in params.items():
                    if key in step.params or not step.params.get(key):
                        step.params[key] = value

        return self._execute_workflow(steps, self.context.current_language)

    def get_available_agents(self) -> list[dict[str, str]]:
        """Get list of available agents with descriptions.

        Returns:
            List of agent info dictionaries
        """
        agents = self.registry.get_all_agents()
        result = []
        for agent_type, agent in agents.items():
            result.append(
                {
                    "type": agent_type,
                    "name": agent.name,
                    "description": agent.description,
                    "description_ko": agent.description_ko,
                    "n_skills": len(agent.skills),
                }
            )
        return result

    def get_agent_skills(self, agent_type: str) -> list[dict[str, Any]]:
        """Get skills for a specific agent.

        Args:
            agent_type: Agent type identifier

        Returns:
            List of skill info dictionaries
        """
        agent = self.registry.get_agent(agent_type)
        if agent is None:
            return []

        result = []
        for skill_name, skill in agent.skills.items():
            result.append(
                {
                    "name": skill_name,
                    "description": skill.description,
                    "description_ko": skill.description_ko,
                    "parameters": skill.parameters,
                    "required_params": skill.required_params,
                }
            )
        return result
