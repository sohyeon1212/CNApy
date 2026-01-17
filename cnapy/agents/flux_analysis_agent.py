"""Flux Analysis Agent for CNApy Multi-Agent System

This agent handles all flux-based analyses:
- FBA (Flux Balance Analysis)
- pFBA (Parsimonious FBA)
- FVA (Flux Variability Analysis)
- MOMA (Linear MOMA)
- ROOM (Regulatory On/Off Minimization)
- Flux Sampling
"""


import cobra

from cnapy.agents.base_agent import (
    BaseAgent,
    Skill,
    SkillResult,
    SkillStatus,
)


class FluxAnalysisAgent(BaseAgent):
    """Agent for flux-based metabolic analyses.

    Handles FBA, pFBA, FVA, MOMA, ROOM, and flux sampling analyses.
    Requires a loaded COBRA model in the context.
    """

    @property
    def name(self) -> str:
        return "flux_analysis"

    @property
    def description(self) -> str:
        return "Performs flux-based metabolic analyses including FBA, pFBA, FVA, MOMA, ROOM, and flux sampling."

    @property
    def description_ko(self) -> str:
        return "FBA, pFBA, FVA, MOMA, ROOM, 플럭스 샘플링을 포함한 플럭스 기반 대사 분석을 수행합니다."

    def _register_skills(self):
        """Register all flux analysis skills."""

        # FBA
        self.register_skill(
            Skill(
                name="perform_fba",
                description="Perform Flux Balance Analysis (FBA) to find optimal flux distribution",
                description_ko="최적 플럭스 분포를 찾기 위한 Flux Balance Analysis (FBA) 수행",
                parameters={},
                required_params=[],
                handler=self._perform_fba,
            )
        )

        # pFBA
        self.register_skill(
            Skill(
                name="perform_pfba",
                description="Perform Parsimonious FBA (pFBA) to find minimum total flux solution",
                description_ko="최소 총 플럭스 솔루션을 찾기 위한 Parsimonious FBA (pFBA) 수행",
                parameters={},
                required_params=[],
                handler=self._perform_pfba,
            )
        )

        # FVA
        self.register_skill(
            Skill(
                name="perform_fva",
                description="Perform Flux Variability Analysis (FVA) to determine flux ranges",
                description_ko="플럭스 범위를 결정하기 위한 Flux Variability Analysis (FVA) 수행",
                parameters={
                    "fraction_of_optimum": {
                        "type": "number",
                        "description": "Fraction of optimal objective to maintain (0.0-1.0)",
                        "default": 0.0,
                    },
                },
                required_params=[],
                handler=self._perform_fva,
            )
        )

        # MOMA
        self.register_skill(
            Skill(
                name="perform_moma",
                description="Perform Linear MOMA (Minimization of Metabolic Adjustment)",
                description_ko="Linear MOMA (대사 조정 최소화) 수행",
                parameters={
                    "reference_fluxes": {
                        "type": "object",
                        "description": "Reference flux values for MOMA comparison",
                        "default": None,
                    },
                },
                required_params=[],
                handler=self._perform_moma,
            )
        )

        # ROOM
        self.register_skill(
            Skill(
                name="perform_room",
                description="Perform ROOM (Regulatory On/Off Minimization) - requires MILP solver",
                description_ko="ROOM (조절 On/Off 최소화) 수행 - MILP 솔버 필요",
                parameters={
                    "reference_fluxes": {
                        "type": "object",
                        "description": "Reference flux values for ROOM comparison",
                        "default": None,
                    },
                    "delta": {
                        "type": "number",
                        "description": "Delta parameter for ROOM",
                        "default": 0.03,
                    },
                    "epsilon": {
                        "type": "number",
                        "description": "Epsilon parameter for ROOM",
                        "default": 0.001,
                    },
                },
                required_params=[],
                handler=self._perform_room,
            )
        )

        # Flux Sampling
        self.register_skill(
            Skill(
                name="perform_flux_sampling",
                description="Perform flux sampling to explore solution space",
                description_ko="솔루션 공간을 탐색하기 위한 플럭스 샘플링 수행",
                parameters={
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of samples to generate",
                        "default": 100,
                    },
                    "thinning": {
                        "type": "integer",
                        "description": "Thinning factor for sampling",
                        "default": 10,
                    },
                    "method": {
                        "type": "string",
                        "description": "Sampling method (optgp or achr)",
                        "default": "optgp",
                    },
                },
                required_params=[],
                handler=self._perform_flux_sampling,
            )
        )

        # Analyze flux distribution
        self.register_skill(
            Skill(
                name="analyze_flux_distribution",
                description="Analyze flux distribution for specific reactions",
                description_ko="특정 반응들의 플럭스 분포 분석",
                parameters={
                    "reaction_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reaction IDs to analyze",
                    },
                },
                required_params=["reaction_ids"],
                handler=self._analyze_flux_distribution,
            )
        )

        # Get objective value
        self.register_skill(
            Skill(
                name="get_objective_value",
                description="Get the current objective value from the last optimization",
                description_ko="마지막 최적화의 현재 목적함수 값 조회",
                parameters={},
                required_params=[],
                handler=self._get_objective_value,
            )
        )

        # Compare flux states
        self.register_skill(
            Skill(
                name="compare_flux_states",
                description="Compare flux values between two states",
                description_ko="두 상태 간의 플럭스 값 비교",
                parameters={
                    "state1": {
                        "type": "object",
                        "description": "First flux state (dict of reaction_id: flux_value)",
                    },
                    "state2": {
                        "type": "object",
                        "description": "Second flux state (dict of reaction_id: flux_value)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum flux difference to report",
                        "default": 0.001,
                    },
                },
                required_params=["state1", "state2"],
                handler=self._compare_flux_states,
            )
        )

    def _check_model(self) -> SkillResult | None:
        """Check if a model is loaded.

        Returns:
            SkillResult with error if no model, None if OK
        """
        if self.context.model is None or len(self.context.model.reactions) == 0:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No model loaded",
                message="No metabolic model is loaded. Please load a model first.",
                message_ko="로드된 대사 모델이 없습니다. 먼저 모델을 로드해주세요.",
            )
        return None

    def _perform_fba(self) -> SkillResult:
        """Perform Flux Balance Analysis."""
        check = self._check_model()
        if check:
            return check

        try:
            from cnapy.core_gui import model_optimization_with_exceptions

            with self.context.model as model:
                # Load scenario into model
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform optimization
                solution = model_optimization_with_exceptions(model)

                # Store solution
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.solution = solution

            if solution.status == "optimal":
                # Update computed values
                if self.context.appdata and self.context.appdata.project:
                    for r, v in solution.fluxes.items():
                        self.context.appdata.project.comp_values[r] = (v, v)
                    self.context.appdata.project.comp_values_type = 0

                obj_value = round(solution.objective_value, 4)
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    data={
                        "objective_value": obj_value,
                        "status": solution.status,
                        "fluxes": solution.fluxes.to_dict(),
                    },
                    message=f"FBA completed successfully. Optimal objective value: {obj_value}",
                    message_ko=f"FBA가 성공적으로 완료되었습니다. 최적 목적함수 값: {obj_value}",
                    metadata={"analysis_type": "fba"},
                )
            elif solution.status == "infeasible":
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    data={"status": solution.status},
                    error="Infeasible",
                    message="The current scenario is infeasible. No solution exists.",
                    message_ko="현재 시나리오가 실행 불가능합니다. 해가 존재하지 않습니다.",
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    data={"status": solution.status},
                    error=f"Solver status: {solution.status}",
                    message=f"Optimization failed with status: {solution.status}",
                    message_ko=f"최적화 실패. 상태: {solution.status}",
                )

        except cobra.exceptions.Infeasible:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="Infeasible",
                message="The current scenario is infeasible.",
                message_ko="현재 시나리오가 실행 불가능합니다.",
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"FBA failed with error: {str(e)}",
                message_ko=f"FBA 실행 중 오류 발생: {str(e)}",
            )

    def _perform_pfba(self) -> SkillResult:
        """Perform Parsimonious FBA."""
        check = self._check_model()
        if check:
            return check

        try:
            with self.context.model as model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform pFBA
                solution = cobra.flux_analysis.pfba(model)

                # Store solution
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.solution = solution

            if solution.status == "optimal":
                # Update computed values
                if self.context.appdata and self.context.appdata.project:
                    for r, v in solution.fluxes.items():
                        self.context.appdata.project.comp_values[r] = (v, v)
                    self.context.appdata.project.comp_values_type = 0

                obj_value = round(solution.objective_value, 4)
                total_flux = round(sum(abs(v) for v in solution.fluxes.values()), 4)

                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    data={
                        "objective_value": obj_value,
                        "total_flux": total_flux,
                        "status": solution.status,
                        "fluxes": solution.fluxes.to_dict(),
                    },
                    message=f"pFBA completed. Objective: {obj_value}, Total flux: {total_flux}",
                    message_ko=f"pFBA 완료. 목적함수: {obj_value}, 총 플럭스: {total_flux}",
                    metadata={"analysis_type": "pfba"},
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    data={"status": solution.status},
                    error=f"Status: {solution.status}",
                    message=f"pFBA failed with status: {solution.status}",
                    message_ko=f"pFBA 실패. 상태: {solution.status}",
                )

        except cobra.exceptions.Infeasible:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="Infeasible",
                message="The current scenario is infeasible.",
                message_ko="현재 시나리오가 실행 불가능합니다.",
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"pFBA failed with error: {str(e)}",
                message_ko=f"pFBA 실행 중 오류 발생: {str(e)}",
            )

    def _perform_fva(self, fraction_of_optimum: float = 0.0) -> SkillResult:
        """Perform Flux Variability Analysis."""
        check = self._check_model()
        if check:
            return check

        try:
            from cobra.flux_analysis import flux_variability_analysis

            with self.context.model as model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform FVA
                fva_result = flux_variability_analysis(
                    model,
                    fraction_of_optimum=fraction_of_optimum,
                )

            # Convert to dict
            minimum = fva_result.minimum.to_dict()
            maximum = fva_result.maximum.to_dict()

            # Update computed values
            if self.context.appdata and self.context.appdata.project:
                for rxn_id in minimum:
                    self.context.appdata.project.comp_values[rxn_id] = (minimum[rxn_id], maximum[rxn_id])
                self.context.appdata.project.fva_values = self.context.appdata.project.comp_values.copy()
                self.context.appdata.project.comp_values_type = 1

            # Count variable reactions
            n_variable = sum(1 for rxn_id in minimum if abs(maximum[rxn_id] - minimum[rxn_id]) > 1e-6)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "minimum": minimum,
                    "maximum": maximum,
                    "fraction_of_optimum": fraction_of_optimum,
                    "n_reactions": len(minimum),
                    "n_variable": n_variable,
                },
                message=f"FVA completed. {n_variable}/{len(minimum)} reactions have variable flux.",
                message_ko=f"FVA 완료. {len(minimum)}개 반응 중 {n_variable}개가 가변 플럭스를 가집니다.",
                metadata={"analysis_type": "fva"},
            )

        except cobra.exceptions.Infeasible:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="Infeasible",
                message="The current scenario is infeasible.",
                message_ko="현재 시나리오가 실행 불가능합니다.",
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"FVA failed with error: {str(e)}",
                message_ko=f"FVA 실행 중 오류 발생: {str(e)}",
            )

    def _perform_moma(self, reference_fluxes: dict[str, float] | None = None) -> SkillResult:
        """Perform Linear MOMA."""
        check = self._check_model()
        if check:
            return check

        # Check for reference fluxes
        if reference_fluxes is None:
            # Try to use previous solution
            if self.context.solution is not None:
                reference_fluxes = self.context.solution.fluxes.to_dict()
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error="No reference fluxes",
                    message="No reference fluxes provided and no previous solution available. Run FBA first.",
                    message_ko="참조 플럭스가 제공되지 않았고 이전 솔루션이 없습니다. 먼저 FBA를 실행하세요.",
                )

        try:
            from cnapy.moma import linear_moma

            with self.context.model as model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform MOMA
                solution = linear_moma(model, reference_fluxes)

                # Store solution
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.solution = solution

            if solution.status == "optimal":
                # Update computed values
                if self.context.appdata and self.context.appdata.project:
                    for r, v in solution.fluxes.items():
                        self.context.appdata.project.comp_values[r] = (v, v)
                    self.context.appdata.project.comp_values_type = 0

                obj_value = round(solution.objective_value, 4)

                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    data={
                        "objective_value": obj_value,
                        "status": solution.status,
                        "fluxes": solution.fluxes.to_dict(),
                    },
                    message=f"MOMA completed. Objective value: {obj_value}",
                    message_ko=f"MOMA 완료. 목적함수 값: {obj_value}",
                    metadata={"analysis_type": "moma"},
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    data={"status": solution.status},
                    error=f"Status: {solution.status}",
                    message=f"MOMA failed with status: {solution.status}",
                    message_ko=f"MOMA 실패. 상태: {solution.status}",
                )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"MOMA failed with error: {str(e)}",
                message_ko=f"MOMA 실행 중 오류 발생: {str(e)}",
            )

    def _perform_room(
        self,
        reference_fluxes: dict[str, float] | None = None,
        delta: float = 0.03,
        epsilon: float = 0.001,
    ) -> SkillResult:
        """Perform ROOM analysis."""
        check = self._check_model()
        if check:
            return check

        # Check for MILP solver
        try:
            from cnapy.moma import has_milp_solver, room

            has_milp, solver_msg = has_milp_solver()
            if not has_milp:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error="No MILP solver",
                    message=f"ROOM requires a MILP solver. {solver_msg}",
                    message_ko=f"ROOM은 MILP 솔버가 필요합니다. {solver_msg}",
                )
        except ImportError:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="ROOM not available",
                message="ROOM functionality is not available.",
                message_ko="ROOM 기능을 사용할 수 없습니다.",
            )

        # Check for reference fluxes
        if reference_fluxes is None:
            if self.context.solution is not None:
                reference_fluxes = self.context.solution.fluxes.to_dict()
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error="No reference fluxes",
                    message="No reference fluxes provided and no previous solution available.",
                    message_ko="참조 플럭스가 제공되지 않았고 이전 솔루션이 없습니다.",
                )

        try:
            with self.context.model as model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform ROOM
                solution = room(model, reference_fluxes, delta=delta, epsilon=epsilon)

                # Store solution
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.solution = solution

            if solution.status == "optimal":
                # Update computed values
                if self.context.appdata and self.context.appdata.project:
                    for r, v in solution.fluxes.items():
                        self.context.appdata.project.comp_values[r] = (v, v)
                    self.context.appdata.project.comp_values_type = 0

                obj_value = round(solution.objective_value, 4)

                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    data={
                        "objective_value": obj_value,
                        "status": solution.status,
                        "fluxes": solution.fluxes.to_dict(),
                    },
                    message=f"ROOM completed. Objective value: {obj_value}",
                    message_ko=f"ROOM 완료. 목적함수 값: {obj_value}",
                    metadata={"analysis_type": "room"},
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    data={"status": solution.status},
                    error=f"Status: {solution.status}",
                    message=f"ROOM failed with status: {solution.status}",
                    message_ko=f"ROOM 실패. 상태: {solution.status}",
                )

        except RuntimeError as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"ROOM failed: {str(e)}",
                message_ko=f"ROOM 실패: {str(e)}",
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"ROOM failed with error: {str(e)}",
                message_ko=f"ROOM 실행 중 오류 발생: {str(e)}",
            )

    def _perform_flux_sampling(
        self,
        n_samples: int = 100,
        thinning: int = 10,
        method: str = "optgp",
    ) -> SkillResult:
        """Perform flux sampling."""
        check = self._check_model()
        if check:
            return check

        try:
            from cnapy.flux_sampling import perform_sampling

            with self.context.model as model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(model)

                # Perform sampling
                samples = perform_sampling(model, n_samples, thinning)

            # Calculate statistics
            mean_fluxes = samples.mean().to_dict()
            std_fluxes = samples.std().to_dict()

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "n_samples": len(samples),
                    "n_reactions": len(samples.columns),
                    "mean_fluxes": mean_fluxes,
                    "std_fluxes": std_fluxes,
                    "samples": samples.to_dict(),
                },
                message=f"Sampling completed. Generated {len(samples)} samples for {len(samples.columns)} reactions.",
                message_ko=f"샘플링 완료. {len(samples.columns)}개 반응에 대해 {len(samples)}개 샘플 생성.",
                metadata={"analysis_type": "sampling"},
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Flux sampling failed with error: {str(e)}",
                message_ko=f"플럭스 샘플링 실행 중 오류 발생: {str(e)}",
            )

    def _analyze_flux_distribution(self, reaction_ids: list[str]) -> SkillResult:
        """Analyze flux distribution for specific reactions."""
        check = self._check_model()
        if check:
            return check

        if not self.context.comp_values:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No flux values",
                message="No flux values available. Run an analysis first.",
                message_ko="사용 가능한 플럭스 값이 없습니다. 먼저 분석을 실행하세요.",
            )

        results = {}
        missing = []

        for rxn_id in reaction_ids:
            if rxn_id in self.context.comp_values:
                flux = self.context.comp_values[rxn_id]
                results[rxn_id] = {
                    "lower": flux[0],
                    "upper": flux[1],
                    "mean": (flux[0] + flux[1]) / 2,
                }
            else:
                missing.append(rxn_id)

        if not results:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No matching reactions",
                message=f"None of the specified reactions were found: {missing}",
                message_ko=f"지정된 반응 중 어느 것도 찾을 수 없습니다: {missing}",
            )

        message_parts = []
        for rxn_id, values in results.items():
            if abs(values["lower"] - values["upper"]) < 1e-6:
                message_parts.append(f"{rxn_id}: {round(values['mean'], 4)}")
            else:
                message_parts.append(f"{rxn_id}: [{round(values['lower'], 4)}, {round(values['upper'], 4)}]")

        return SkillResult(
            status=SkillStatus.SUCCESS if not missing else SkillStatus.PARTIAL,
            data={
                "fluxes": results,
                "missing": missing,
            },
            message="Flux distribution: " + ", ".join(message_parts),
            message_ko="플럭스 분포: " + ", ".join(message_parts),
        )

    def _get_objective_value(self) -> SkillResult:
        """Get the current objective value."""
        if self.context.solution is None:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No solution",
                message="No optimization solution available. Run FBA first.",
                message_ko="사용 가능한 최적화 솔루션이 없습니다. 먼저 FBA를 실행하세요.",
            )

        obj_value = round(self.context.solution.objective_value, 4)
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "objective_value": obj_value,
                "status": self.context.solution.status,
            },
            message=f"Current objective value: {obj_value}",
            message_ko=f"현재 목적함수 값: {obj_value}",
        )

    def _compare_flux_states(
        self,
        state1: dict[str, float],
        state2: dict[str, float],
        threshold: float = 0.001,
    ) -> SkillResult:
        """Compare flux values between two states."""
        all_reactions = set(state1.keys()) | set(state2.keys())

        differences = []
        for rxn_id in all_reactions:
            v1 = state1.get(rxn_id, 0.0)
            v2 = state2.get(rxn_id, 0.0)
            diff = v2 - v1

            if abs(diff) > threshold:
                differences.append(
                    {
                        "reaction_id": rxn_id,
                        "state1": v1,
                        "state2": v2,
                        "difference": diff,
                        "fold_change": v2 / v1 if abs(v1) > 1e-10 else float("inf"),
                    }
                )

        # Sort by absolute difference
        differences.sort(key=lambda x: abs(x["difference"]), reverse=True)

        n_changed = len(differences)
        top_changes = differences[:10]

        message_parts = [f"{n_changed} reactions changed significantly."]
        if top_changes:
            message_parts.append("Top changes:")
            for d in top_changes[:5]:
                message_parts.append(f"  {d['reaction_id']}: {round(d['difference'], 4)}")

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "n_total_reactions": len(all_reactions),
                "n_changed": n_changed,
                "differences": differences,
                "threshold": threshold,
            },
            message=" ".join(message_parts),
            message_ko=f"{n_changed}개 반응이 유의미하게 변화했습니다.",
        )
