"""Scenario Manager Agent for CNApy Multi-Agent System

This agent handles scenario and condition management:
- Apply predefined culture conditions (aerobic, anaerobic, etc.)
- Set carbon/nitrogen sources
- Set reaction bounds
- Save/load scenarios
- Manage objectives
"""

import os

from cnapy.agents.base_agent import (
    BaseAgent,
    Skill,
    SkillResult,
    SkillStatus,
)
from cnapy.agents.predefined_scenarios import (
    CARBON_SOURCES,
    CULTURE_CONDITIONS,
    NITROGEN_SOURCES,
    OXYGEN_EXCHANGE_ALTERNATIVES,
    get_carbon_source,
    get_culture_condition,
    get_nitrogen_source,
    list_carbon_sources,
    list_conditions,
    list_nitrogen_sources,
)


class ScenarioManagerAgent(BaseAgent):
    """Agent for managing culture conditions and scenarios.

    Handles predefined conditions, reaction bounds, carbon/nitrogen sources,
    objective functions, and scenario save/load operations.
    """

    @property
    def name(self) -> str:
        return "scenario"

    @property
    def description(self) -> str:
        return "Manages culture conditions, scenarios, reaction bounds, and objective functions."

    @property
    def description_ko(self) -> str:
        return "배양 조건, 시나리오, 반응 경계, 목적함수를 관리합니다."

    def _register_skills(self):
        """Register all scenario management skills."""

        # Apply predefined condition
        self.register_skill(
            Skill(
                name="apply_condition",
                description="Apply a predefined culture condition (aerobic, anaerobic, microaerobic, etc.)",
                description_ko="사전정의된 배양 조건 적용 (호기성, 혐기성, 미호기성 등)",
                parameters={
                    "condition_name": {
                        "type": "string",
                        "description": "Name of condition (aerobic, anaerobic, etc.)",
                        "enum": list(CULTURE_CONDITIONS.keys()),
                    },
                },
                required_params=["condition_name"],
                handler=self._apply_condition,
            )
        )

        # Set reaction bounds
        self.register_skill(
            Skill(
                name="set_reaction_bounds",
                description="Set lower and upper bounds for a reaction",
                description_ko="반응의 하한/상한 경계 설정",
                parameters={
                    "reaction_id": {
                        "type": "string",
                        "description": "Reaction ID to modify",
                    },
                    "lower_bound": {
                        "type": "number",
                        "description": "Lower bound value",
                    },
                    "upper_bound": {
                        "type": "number",
                        "description": "Upper bound value",
                    },
                },
                required_params=["reaction_id", "lower_bound", "upper_bound"],
                handler=self._set_reaction_bounds,
            )
        )

        # Set carbon source
        self.register_skill(
            Skill(
                name="set_carbon_source",
                description="Set the primary carbon source and uptake rate",
                description_ko="주요 탄소원 및 섭취율 설정",
                parameters={
                    "carbon_source": {
                        "type": "string",
                        "description": "Carbon source name (glucose, xylose, glycerol, acetate, etc.)",
                        "enum": list(CARBON_SOURCES.keys()),
                    },
                    "uptake_rate": {
                        "type": "number",
                        "description": "Uptake rate (negative for consumption)",
                        "default": -10.0,
                    },
                },
                required_params=["carbon_source"],
                handler=self._set_carbon_source,
            )
        )

        # Set nitrogen source
        self.register_skill(
            Skill(
                name="set_nitrogen_source",
                description="Set the primary nitrogen source and uptake rate",
                description_ko="주요 질소원 및 섭취율 설정",
                parameters={
                    "nitrogen_source": {
                        "type": "string",
                        "description": "Nitrogen source name (ammonium, nitrate, glutamate, glutamine)",
                        "enum": list(NITROGEN_SOURCES.keys()),
                    },
                    "uptake_rate": {
                        "type": "number",
                        "description": "Uptake rate (negative for consumption)",
                        "default": -10.0,
                    },
                },
                required_params=["nitrogen_source"],
                handler=self._set_nitrogen_source,
            )
        )

        # Set objective
        self.register_skill(
            Skill(
                name="set_objective",
                description="Set the optimization objective function",
                description_ko="최적화 목적함수 설정",
                parameters={
                    "reaction_id": {
                        "type": "string",
                        "description": "Reaction ID to use as objective",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Optimization direction (max or min)",
                        "enum": ["max", "min"],
                        "default": "max",
                    },
                },
                required_params=["reaction_id"],
                handler=self._set_objective,
            )
        )

        # Get current scenario
        self.register_skill(
            Skill(
                name="get_current_scenario",
                description="Get the current scenario settings",
                description_ko="현재 시나리오 설정 조회",
                parameters={},
                required_params=[],
                handler=self._get_current_scenario,
            )
        )

        # Clear scenario
        self.register_skill(
            Skill(
                name="clear_scenario",
                description="Clear all scenario flux values",
                description_ko="모든 시나리오 플럭스 값 초기화",
                parameters={},
                required_params=[],
                handler=self._clear_scenario,
            )
        )

        # Save scenario
        self.register_skill(
            Skill(
                name="save_scenario",
                description="Save the current scenario to a file",
                description_ko="현재 시나리오를 파일로 저장",
                parameters={
                    "filename": {
                        "type": "string",
                        "description": "File name for the scenario",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the scenario",
                        "default": "",
                    },
                },
                required_params=["filename"],
                handler=self._save_scenario,
            )
        )

        # Load scenario
        self.register_skill(
            Skill(
                name="load_scenario",
                description="Load a scenario from a file",
                description_ko="파일에서 시나리오 로드",
                parameters={
                    "filename": {
                        "type": "string",
                        "description": "File name of the scenario to load",
                    },
                },
                required_params=["filename"],
                handler=self._load_scenario,
            )
        )

        # List available conditions
        self.register_skill(
            Skill(
                name="list_conditions",
                description="List all available predefined culture conditions",
                description_ko="사용 가능한 모든 사전정의 배양 조건 목록",
                parameters={},
                required_params=[],
                handler=self._list_conditions,
            )
        )

        # List carbon sources
        self.register_skill(
            Skill(
                name="list_carbon_sources",
                description="List all available carbon sources",
                description_ko="사용 가능한 모든 탄소원 목록",
                parameters={},
                required_params=[],
                handler=self._list_carbon_sources,
            )
        )

        # List nitrogen sources
        self.register_skill(
            Skill(
                name="list_nitrogen_sources",
                description="List all available nitrogen sources",
                description_ko="사용 가능한 모든 질소원 목록",
                parameters={},
                required_params=[],
                handler=self._list_nitrogen_sources,
            )
        )

    def _check_model(self) -> SkillResult | None:
        """Check if a model is loaded."""
        if self.context.model is None or len(self.context.model.reactions) == 0:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No model loaded",
                message="No metabolic model is loaded. Please load a model first.",
                message_ko="로드된 대사 모델이 없습니다. 먼저 모델을 로드해주세요.",
            )
        return None

    def _find_reaction_id(self, patterns: list[str]) -> str | None:
        """Find a reaction ID from a list of patterns."""
        if self.context.model is None:
            return None
        reaction_ids = {r.id for r in self.context.model.reactions}
        for pattern in patterns:
            if pattern in reaction_ids:
                return pattern
        return None

    def _apply_condition(self, condition_name: str) -> SkillResult:
        """Apply a predefined culture condition."""
        check = self._check_model()
        if check:
            return check

        condition = get_culture_condition(condition_name)
        if condition is None:
            available = ", ".join(CULTURE_CONDITIONS.keys())
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Unknown condition: {condition_name}",
                message=f"Unknown condition '{condition_name}'. Available: {available}",
                message_ko=f"알 수 없는 조건 '{condition_name}'. 사용 가능: {available}",
            )

        applied = []
        not_found = []

        for rxn_pattern, bounds in condition.bounds.items():
            # Try to find the reaction with alternatives
            if rxn_pattern.startswith("EX_o2"):
                alternatives = OXYGEN_EXCHANGE_ALTERNATIVES
            else:
                alternatives = [rxn_pattern]

            rxn_id = self._find_reaction_id(alternatives)

            if rxn_id:
                # Apply bounds to scenario
                if self.context.appdata:
                    self.context.appdata.scen_values_set(rxn_id, bounds)
                # Also update model bounds directly for immediate effect
                if self.context.model and rxn_id in self.context.model.reactions:
                    rxn = self.context.model.reactions.get_by_id(rxn_id)
                    rxn.lower_bound = bounds[0]
                    rxn.upper_bound = bounds[1]
                applied.append(rxn_id)
            else:
                not_found.append(rxn_pattern)

        if applied:
            status = SkillStatus.SUCCESS if not not_found else SkillStatus.PARTIAL
            message = f"Applied '{condition.display_name}' condition. Modified: {', '.join(applied)}"
            message_ko = f"'{condition.display_name_ko}' 조건 적용됨. 수정된 반응: {', '.join(applied)}"

            if not_found:
                message += f". Not found: {', '.join(not_found)}"
                message_ko += f". 찾지 못함: {', '.join(not_found)}"

            return SkillResult(
                status=status,
                data={
                    "condition": condition_name,
                    "applied": applied,
                    "not_found": not_found,
                },
                message=message,
                message_ko=message_ko,
            )
        else:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No reactions found",
                message=f"Could not find any reactions for condition '{condition_name}'.",
                message_ko=f"'{condition_name}' 조건에 해당하는 반응을 찾을 수 없습니다.",
            )

    def _set_reaction_bounds(
        self,
        reaction_id: str,
        lower_bound: float,
        upper_bound: float,
    ) -> SkillResult:
        """Set bounds for a specific reaction."""
        check = self._check_model()
        if check:
            return check

        # Check if reaction exists
        if reaction_id not in [r.id for r in self.context.model.reactions]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Reaction not found: {reaction_id}",
                message=f"Reaction '{reaction_id}' not found in the model.",
                message_ko=f"반응 '{reaction_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        # Apply bounds to scenario
        if self.context.appdata:
            self.context.appdata.scen_values_set(reaction_id, (lower_bound, upper_bound))

        # Also update model bounds directly for immediate effect
        if self.context.model:
            rxn = self.context.model.reactions.get_by_id(reaction_id)
            rxn.lower_bound = lower_bound
            rxn.upper_bound = upper_bound

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "reaction_id": reaction_id,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            message=f"Set bounds for {reaction_id}: [{lower_bound}, {upper_bound}]",
            message_ko=f"{reaction_id}의 경계 설정됨: [{lower_bound}, {upper_bound}]",
        )

    def _set_carbon_source(
        self,
        carbon_source: str,
        uptake_rate: float = -10.0,
    ) -> SkillResult:
        """Set the carbon source."""
        check = self._check_model()
        if check:
            return check

        source = get_carbon_source(carbon_source)
        if source is None:
            available = ", ".join(CARBON_SOURCES.keys())
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Unknown carbon source: {carbon_source}",
                message=f"Unknown carbon source '{carbon_source}'. Available: {available}",
                message_ko=f"알 수 없는 탄소원 '{carbon_source}'. 사용 가능: {available}",
            )

        # Find the exchange reaction
        alternatives = [source.exchange_reaction] + source.alternatives
        rxn_id = self._find_reaction_id(alternatives)

        if rxn_id is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="Exchange reaction not found",
                message=f"Could not find exchange reaction for {carbon_source}.",
                message_ko=f"{carbon_source}의 교환 반응을 찾을 수 없습니다.",
            )

        # Apply bounds (uptake_rate is negative, secretion is 0)
        if self.context.appdata:
            self.context.appdata.scen_values_set(rxn_id, (uptake_rate, 0.0))

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "carbon_source": carbon_source,
                "reaction_id": rxn_id,
                "uptake_rate": uptake_rate,
            },
            message=f"Set {source.display_name} as carbon source with uptake rate {uptake_rate}",
            message_ko=f"{source.display_name_ko}을(를) 탄소원으로 설정 (섭취율: {uptake_rate})",
        )

    def _set_nitrogen_source(
        self,
        nitrogen_source: str,
        uptake_rate: float = -10.0,
    ) -> SkillResult:
        """Set the nitrogen source."""
        check = self._check_model()
        if check:
            return check

        source = get_nitrogen_source(nitrogen_source)
        if source is None:
            available = ", ".join(NITROGEN_SOURCES.keys())
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Unknown nitrogen source: {nitrogen_source}",
                message=f"Unknown nitrogen source '{nitrogen_source}'. Available: {available}",
                message_ko=f"알 수 없는 질소원 '{nitrogen_source}'. 사용 가능: {available}",
            )

        # Find the exchange reaction
        alternatives = [source.exchange_reaction] + source.alternatives
        rxn_id = self._find_reaction_id(alternatives)

        if rxn_id is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="Exchange reaction not found",
                message=f"Could not find exchange reaction for {nitrogen_source}.",
                message_ko=f"{nitrogen_source}의 교환 반응을 찾을 수 없습니다.",
            )

        # Apply bounds
        if self.context.appdata:
            self.context.appdata.scen_values_set(rxn_id, (uptake_rate, 1000.0))

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "nitrogen_source": nitrogen_source,
                "reaction_id": rxn_id,
                "uptake_rate": uptake_rate,
            },
            message=f"Set {source.display_name} as nitrogen source with uptake rate {uptake_rate}",
            message_ko=f"{source.display_name_ko}을(를) 질소원으로 설정 (섭취율: {uptake_rate})",
        )

    def _set_objective(
        self,
        reaction_id: str,
        direction: str = "max",
    ) -> SkillResult:
        """Set the optimization objective."""
        check = self._check_model()
        if check:
            return check

        # Check if reaction exists
        if reaction_id not in [r.id for r in self.context.model.reactions]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Reaction not found: {reaction_id}",
                message=f"Reaction '{reaction_id}' not found in the model.",
                message_ko=f"반응 '{reaction_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        # Set objective
        self.context.model.objective = reaction_id
        self.context.model.objective_direction = direction

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "reaction_id": reaction_id,
                "direction": direction,
            },
            message=f"Set objective to {direction}imize {reaction_id}",
            message_ko=f"목적함수 설정: {reaction_id} {direction}imize",
        )

    def _get_current_scenario(self) -> SkillResult:
        """Get the current scenario settings."""
        check = self._check_model()
        if check:
            return check

        if self.context.scenario is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No scenario",
                message="No scenario is currently loaded.",
                message_ko="현재 로드된 시나리오가 없습니다.",
            )

        scenario = self.context.scenario
        flux_values = dict(scenario)
        n_values = len(flux_values)

        # Get objective info
        obj_rxn = str(self.context.model.objective.expression) if self.context.model.objective else "None"
        obj_dir = (
            self.context.model.objective_direction if hasattr(self.context.model, "objective_direction") else "max"
        )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "flux_values": flux_values,
                "n_flux_values": n_values,
                "pinned_reactions": list(scenario.pinned_reactions) if hasattr(scenario, "pinned_reactions") else [],
                "description": scenario.description if hasattr(scenario, "description") else "",
                "objective": obj_rxn,
                "objective_direction": obj_dir,
            },
            message=f"Current scenario has {n_values} flux values set.",
            message_ko=f"현재 시나리오에 {n_values}개의 플럭스 값이 설정되어 있습니다.",
        )

    def _clear_scenario(self) -> SkillResult:
        """Clear all scenario flux values."""
        if self.context.appdata:
            self.context.appdata.scen_values_clear()

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={},
            message="Scenario cleared. All flux values have been reset.",
            message_ko="시나리오 초기화됨. 모든 플럭스 값이 리셋되었습니다.",
        )

    def _save_scenario(
        self,
        filename: str,
        description: str = "",
    ) -> SkillResult:
        """Save the current scenario to a file."""
        if self.context.scenario is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No scenario",
                message="No scenario to save.",
                message_ko="저장할 시나리오가 없습니다.",
            )

        # Ensure .scen extension
        if not filename.endswith(".scen"):
            filename += ".scen"

        # Set description
        if description:
            self.context.scenario.description = description

        try:
            # Determine full path
            if self.context.appdata:
                work_dir = self.context.appdata.work_directory
            else:
                work_dir = os.getcwd()

            if not os.path.isabs(filename):
                filepath = os.path.join(work_dir, filename)
            else:
                filepath = filename

            self.context.scenario.save(filepath)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "filename": filepath,
                    "description": description,
                },
                message=f"Scenario saved to {filepath}",
                message_ko=f"시나리오가 {filepath}에 저장되었습니다.",
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Failed to save scenario: {str(e)}",
                message_ko=f"시나리오 저장 실패: {str(e)}",
            )

    def _load_scenario(self, filename: str) -> SkillResult:
        """Load a scenario from a file."""
        check = self._check_model()
        if check:
            return check

        # Determine full path
        if self.context.appdata:
            work_dir = self.context.appdata.work_directory
        else:
            work_dir = os.getcwd()

        if not os.path.isabs(filename):
            filepath = os.path.join(work_dir, filename)
        else:
            filepath = filename

        if not os.path.exists(filepath):
            # Try with .scen extension
            if not filepath.endswith(".scen"):
                filepath += ".scen"
            if not os.path.exists(filepath):
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error="File not found",
                    message=f"Scenario file not found: {filename}",
                    message_ko=f"시나리오 파일을 찾을 수 없습니다: {filename}",
                )

        try:
            unknown_ids, incompatible, skipped = self.context.scenario.load(filepath, self.context.appdata)

            message = f"Scenario loaded from {filepath}."
            message_ko = f"{filepath}에서 시나리오를 로드했습니다."

            if unknown_ids:
                message += f" Unknown IDs: {len(unknown_ids)}"
                message_ko += f" 알 수 없는 ID: {len(unknown_ids)}"

            status = SkillStatus.SUCCESS if not (unknown_ids or incompatible) else SkillStatus.PARTIAL

            return SkillResult(
                status=status,
                data={
                    "filename": filepath,
                    "unknown_ids": unknown_ids,
                    "incompatible_constraints": incompatible,
                    "skipped_reactions": skipped,
                },
                message=message,
                message_ko=message_ko,
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Failed to load scenario: {str(e)}",
                message_ko=f"시나리오 로드 실패: {str(e)}",
            )

    def _list_conditions(self) -> SkillResult:
        """List available culture conditions."""
        conditions = list_conditions()
        formatted = []
        for name, display, display_ko in conditions:
            formatted.append(
                {
                    "name": name,
                    "display_name": display,
                    "display_name_ko": display_ko,
                }
            )

        names = [c["name"] for c in formatted]
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={"conditions": formatted},
            message=f"Available conditions: {', '.join(names)}",
            message_ko=f"사용 가능한 조건: {', '.join(names)}",
        )

    def _list_carbon_sources(self) -> SkillResult:
        """List available carbon sources."""
        sources = list_carbon_sources()
        formatted = []
        for name, display, display_ko in sources:
            formatted.append(
                {
                    "name": name,
                    "display_name": display,
                    "display_name_ko": display_ko,
                }
            )

        names = [s["name"] for s in formatted]
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={"carbon_sources": formatted},
            message=f"Available carbon sources: {', '.join(names)}",
            message_ko=f"사용 가능한 탄소원: {', '.join(names)}",
        )

    def _list_nitrogen_sources(self) -> SkillResult:
        """List available nitrogen sources."""
        sources = list_nitrogen_sources()
        formatted = []
        for name, display, display_ko in sources:
            formatted.append(
                {
                    "name": name,
                    "display_name": display,
                    "display_name_ko": display_ko,
                }
            )

        names = [s["name"] for s in formatted]
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={"nitrogen_sources": formatted},
            message=f"Available nitrogen sources: {', '.join(names)}",
            message_ko=f"사용 가능한 질소원: {', '.join(names)}",
        )
