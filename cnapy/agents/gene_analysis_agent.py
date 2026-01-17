"""Gene Analysis Agent for CNApy Multi-Agent System

This agent handles gene-related analyses:
- Single/multiple gene knockouts
- Essential gene finding
- Essential reaction finding
- Gene knockout scans
- Synthetic lethal pair finding
"""

from typing import Any, Dict, List, Optional
import cobra

from cnapy.agents.base_agent import (
    BaseAgent,
    AgentContext,
    Skill,
    SkillResult,
    SkillStatus,
)


class GeneAnalysisAgent(BaseAgent):
    """Agent for gene-related metabolic analyses.

    Handles gene knockouts, essential gene detection, and related analyses.
    """

    @property
    def name(self) -> str:
        return "gene_analysis"

    @property
    def description(self) -> str:
        return "Performs gene knockout analyses, finds essential genes, and analyzes gene-reaction relationships."

    @property
    def description_ko(self) -> str:
        return "유전자 녹아웃 분석, 필수 유전자 탐색, 유전자-반응 관계 분석을 수행합니다."

    def _register_skills(self):
        """Register all gene analysis skills."""

        # Single gene knockout
        self.register_skill(Skill(
            name="knockout_gene",
            description="Simulate a single gene knockout and analyze the effect",
            description_ko="단일 유전자 녹아웃을 시뮬레이션하고 효과 분석",
            parameters={
                "gene_id": {
                    "type": "string",
                    "description": "Gene ID to knockout",
                },
            },
            required_params=["gene_id"],
            handler=self._knockout_gene,
        ))

        # Multiple gene knockout
        self.register_skill(Skill(
            name="knockout_genes",
            description="Simulate multiple gene knockouts simultaneously",
            description_ko="다중 유전자 동시 녹아웃 시뮬레이션",
            parameters={
                "gene_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene IDs to knockout",
                },
            },
            required_params=["gene_ids"],
            handler=self._knockout_genes,
        ))

        # Find essential genes
        self.register_skill(Skill(
            name="find_essential_genes",
            description="Find genes whose knockout leads to zero or near-zero growth",
            description_ko="녹아웃 시 성장이 없거나 거의 없는 필수 유전자 탐색",
            parameters={
                "threshold": {
                    "type": "number",
                    "description": "Growth rate threshold below which a gene is considered essential (default: 0.01)",
                    "default": 0.01,
                },
            },
            required_params=[],
            handler=self._find_essential_genes,
        ))

        # Find essential reactions
        self.register_skill(Skill(
            name="find_essential_reactions",
            description="Find reactions whose removal leads to zero or near-zero growth",
            description_ko="제거 시 성장이 없거나 거의 없는 필수 반응 탐색",
            parameters={
                "threshold": {
                    "type": "number",
                    "description": "Growth rate threshold below which a reaction is considered essential",
                    "default": 0.01,
                },
            },
            required_params=[],
            handler=self._find_essential_reactions,
        ))

        # Single gene knockout scan
        self.register_skill(Skill(
            name="single_gene_ko_scan",
            description="Perform single gene knockout scan for all genes",
            description_ko="모든 유전자에 대한 단일 유전자 녹아웃 스캔 수행",
            parameters={},
            required_params=[],
            handler=self._single_gene_ko_scan,
        ))

        # Get gene info
        self.register_skill(Skill(
            name="get_gene_info",
            description="Get information about a specific gene",
            description_ko="특정 유전자 정보 조회",
            parameters={
                "gene_id": {
                    "type": "string",
                    "description": "Gene ID to query",
                },
            },
            required_params=["gene_id"],
            handler=self._get_gene_info,
        ))

        # Analyze gene impact
        self.register_skill(Skill(
            name="analyze_gene_impact",
            description="Analyze the metabolic impact of a gene knockout",
            description_ko="유전자 녹아웃의 대사적 영향 분석",
            parameters={
                "gene_id": {
                    "type": "string",
                    "description": "Gene ID to analyze",
                },
            },
            required_params=["gene_id"],
            handler=self._analyze_gene_impact,
        ))

    def _check_model(self) -> Optional[SkillResult]:
        """Check if a model is loaded."""
        if self.context.model is None or len(self.context.model.reactions) == 0:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No model loaded",
                message="No metabolic model is loaded. Please load a model first.",
                message_ko="로드된 대사 모델이 없습니다. 먼저 모델을 로드해주세요.",
            )
        return None

    def _knockout_gene(self, gene_id: str) -> SkillResult:
        """Simulate a single gene knockout."""
        check = self._check_model()
        if check:
            return check

        try:
            # Check if gene exists
            model = self.context.model
            if gene_id not in [g.id for g in model.genes]:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error=f"Gene not found: {gene_id}",
                    message=f"Gene '{gene_id}' not found in the model.",
                    message_ko=f"유전자 '{gene_id}'을(를) 모델에서 찾을 수 없습니다.",
                )

            # Perform knockout simulation
            with model as ko_model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(ko_model)

                # Get wild-type growth first
                wt_solution = ko_model.optimize()
                wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

                # Find associated reactions and their GPR rules
                gene = ko_model.genes.get_by_id(gene_id)
                affected_reactions = list(gene.reactions)

                # Knockout the gene
                ko_model.genes.get_by_id(gene_id).knock_out()

                # Optimize
                ko_solution = ko_model.optimize()
                ko_growth = ko_solution.objective_value if ko_solution.status == "optimal" else 0.0

            # Calculate growth ratio
            if wt_growth > 1e-6:
                growth_ratio = ko_growth / wt_growth
            else:
                growth_ratio = 0.0 if ko_growth < 1e-6 else float("inf")

            is_essential = growth_ratio < 0.01

            # Prepare affected reactions info
            affected_rxn_info = [
                {"id": r.id, "name": r.name}
                for r in affected_reactions
            ]

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "gene_id": gene_id,
                    "wt_growth": round(wt_growth, 4),
                    "ko_growth": round(ko_growth, 4),
                    "growth_ratio": round(growth_ratio, 4),
                    "is_essential": is_essential,
                    "affected_reactions": affected_rxn_info,
                },
                message=(
                    f"Gene {gene_id} knockout: Growth {round(wt_growth, 4)} → {round(ko_growth, 4)} "
                    f"({round(growth_ratio * 100, 1)}%). "
                    f"{'Essential!' if is_essential else 'Non-essential.'} "
                    f"Affects {len(affected_reactions)} reaction(s)."
                ),
                message_ko=(
                    f"유전자 {gene_id} 녹아웃: 성장률 {round(wt_growth, 4)} → {round(ko_growth, 4)} "
                    f"({round(growth_ratio * 100, 1)}%). "
                    f"{'필수 유전자!' if is_essential else '비필수 유전자.'} "
                    f"{len(affected_reactions)}개 반응에 영향."
                ),
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Gene knockout failed: {str(e)}",
                message_ko=f"유전자 녹아웃 실패: {str(e)}",
            )

    def _knockout_genes(self, gene_ids: List[str]) -> SkillResult:
        """Simulate multiple gene knockouts."""
        check = self._check_model()
        if check:
            return check

        try:
            model = self.context.model

            # Validate all genes exist
            model_gene_ids = {g.id for g in model.genes}
            missing = [g for g in gene_ids if g not in model_gene_ids]
            if missing:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    error=f"Genes not found: {missing}",
                    message=f"Some genes not found: {', '.join(missing)}",
                    message_ko=f"일부 유전자를 찾을 수 없습니다: {', '.join(missing)}",
                )

            with model as ko_model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(ko_model)

                # Get wild-type growth
                wt_solution = ko_model.optimize()
                wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

                # Knockout all genes
                affected_reactions = set()
                for gene_id in gene_ids:
                    gene = ko_model.genes.get_by_id(gene_id)
                    affected_reactions.update(gene.reactions)
                    gene.knock_out()

                # Optimize
                ko_solution = ko_model.optimize()
                ko_growth = ko_solution.objective_value if ko_solution.status == "optimal" else 0.0

            # Calculate growth ratio
            if wt_growth > 1e-6:
                growth_ratio = ko_growth / wt_growth
            else:
                growth_ratio = 0.0 if ko_growth < 1e-6 else float("inf")

            is_lethal = growth_ratio < 0.01

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "gene_ids": gene_ids,
                    "n_genes": len(gene_ids),
                    "wt_growth": round(wt_growth, 4),
                    "ko_growth": round(ko_growth, 4),
                    "growth_ratio": round(growth_ratio, 4),
                    "is_lethal": is_lethal,
                    "affected_reactions": len(affected_reactions),
                },
                message=(
                    f"Multi-gene knockout ({len(gene_ids)} genes): "
                    f"Growth {round(wt_growth, 4)} → {round(ko_growth, 4)} ({round(growth_ratio * 100, 1)}%). "
                    f"{'Lethal combination!' if is_lethal else 'Viable.'}"
                ),
                message_ko=(
                    f"다중 유전자 녹아웃 ({len(gene_ids)}개): "
                    f"성장률 {round(wt_growth, 4)} → {round(ko_growth, 4)} ({round(growth_ratio * 100, 1)}%). "
                    f"{'치사 조합!' if is_lethal else '생존 가능.'}"
                ),
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Multi-gene knockout failed: {str(e)}",
                message_ko=f"다중 유전자 녹아웃 실패: {str(e)}",
            )

    def _find_essential_genes(self, threshold: float = 0.01) -> SkillResult:
        """Find essential genes."""
        check = self._check_model()
        if check:
            return check

        try:
            from cobra.flux_analysis import single_gene_deletion

            model = self.context.model

            with model as test_model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(test_model)

                # Get wild-type growth
                wt_solution = test_model.optimize()
                wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

                if wt_growth < 1e-6:
                    return SkillResult(
                        status=SkillStatus.FAILURE,
                        error="No growth",
                        message="Wild-type has no growth. Cannot determine essential genes.",
                        message_ko="야생형 성장이 없습니다. 필수 유전자를 결정할 수 없습니다.",
                    )

                # Perform single gene deletion
                deletion_results = single_gene_deletion(test_model)

            # Identify essential genes
            essential_genes = []
            non_essential_genes = []

            for idx, row in deletion_results.iterrows():
                gene_id = list(idx)[0] if isinstance(idx, frozenset) else idx
                growth = row["growth"]

                if growth is None or growth < threshold * wt_growth:
                    essential_genes.append({
                        "gene_id": gene_id,
                        "growth": round(growth, 6) if growth is not None else 0.0,
                    })
                else:
                    non_essential_genes.append(gene_id)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "wt_growth": round(wt_growth, 4),
                    "threshold": threshold,
                    "n_total_genes": len(model.genes),
                    "n_essential": len(essential_genes),
                    "n_non_essential": len(non_essential_genes),
                    "essential_genes": essential_genes,
                },
                message=(
                    f"Found {len(essential_genes)} essential genes out of {len(model.genes)} total genes "
                    f"(threshold: {threshold * 100}% of WT growth {round(wt_growth, 4)})."
                ),
                message_ko=(
                    f"총 {len(model.genes)}개 유전자 중 {len(essential_genes)}개 필수 유전자 발견 "
                    f"(임계값: WT 성장률 {round(wt_growth, 4)}의 {threshold * 100}%)."
                ),
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Essential gene analysis failed: {str(e)}",
                message_ko=f"필수 유전자 분석 실패: {str(e)}",
            )

    def _find_essential_reactions(self, threshold: float = 0.01) -> SkillResult:
        """Find essential reactions."""
        check = self._check_model()
        if check:
            return check

        try:
            from cobra.flux_analysis import single_reaction_deletion

            model = self.context.model

            with model as test_model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(test_model)

                # Get wild-type growth
                wt_solution = test_model.optimize()
                wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

                if wt_growth < 1e-6:
                    return SkillResult(
                        status=SkillStatus.FAILURE,
                        error="No growth",
                        message="Wild-type has no growth. Cannot determine essential reactions.",
                        message_ko="야생형 성장이 없습니다. 필수 반응을 결정할 수 없습니다.",
                    )

                # Perform single reaction deletion
                deletion_results = single_reaction_deletion(test_model)

            # Identify essential reactions
            essential_reactions = []

            for idx, row in deletion_results.iterrows():
                rxn_id = list(idx)[0] if isinstance(idx, frozenset) else idx
                growth = row["growth"]

                if growth is None or growth < threshold * wt_growth:
                    essential_reactions.append({
                        "reaction_id": rxn_id,
                        "growth": round(growth, 6) if growth is not None else 0.0,
                    })

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "wt_growth": round(wt_growth, 4),
                    "threshold": threshold,
                    "n_total_reactions": len(model.reactions),
                    "n_essential": len(essential_reactions),
                    "essential_reactions": essential_reactions,
                },
                message=(
                    f"Found {len(essential_reactions)} essential reactions out of {len(model.reactions)} total."
                ),
                message_ko=(
                    f"총 {len(model.reactions)}개 반응 중 {len(essential_reactions)}개 필수 반응 발견."
                ),
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Essential reaction analysis failed: {str(e)}",
                message_ko=f"필수 반응 분석 실패: {str(e)}",
            )

    def _single_gene_ko_scan(self) -> SkillResult:
        """Perform single gene knockout scan."""
        check = self._check_model()
        if check:
            return check

        try:
            from cobra.flux_analysis import single_gene_deletion

            model = self.context.model

            with model as test_model:
                # Load scenario
                if self.context.appdata and self.context.appdata.project:
                    self.context.appdata.project.load_scenario_into_model(test_model)

                # Get wild-type growth
                wt_solution = test_model.optimize()
                wt_growth = wt_solution.objective_value if wt_solution.status == "optimal" else 0.0

                # Perform deletion scan
                deletion_results = single_gene_deletion(test_model)

            # Process results
            results = []
            for idx, row in deletion_results.iterrows():
                gene_id = list(idx)[0] if isinstance(idx, frozenset) else idx
                growth = row["growth"]
                ratio = (growth / wt_growth) if wt_growth > 1e-6 and growth is not None else 0.0

                results.append({
                    "gene_id": gene_id,
                    "growth": round(growth, 6) if growth is not None else 0.0,
                    "growth_ratio": round(ratio, 4),
                })

            # Sort by growth ratio (lowest first)
            results.sort(key=lambda x: x["growth_ratio"])

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={
                    "wt_growth": round(wt_growth, 4),
                    "n_genes": len(results),
                    "results": results,
                },
                message=f"Completed single gene knockout scan for {len(results)} genes.",
                message_ko=f"{len(results)}개 유전자에 대한 단일 유전자 녹아웃 스캔 완료.",
            )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=str(e),
                message=f"Gene knockout scan failed: {str(e)}",
                message_ko=f"유전자 녹아웃 스캔 실패: {str(e)}",
            )

    def _get_gene_info(self, gene_id: str) -> SkillResult:
        """Get information about a specific gene."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        # Check if gene exists
        if gene_id not in [g.id for g in model.genes]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Gene not found: {gene_id}",
                message=f"Gene '{gene_id}' not found in the model.",
                message_ko=f"유전자 '{gene_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        gene = model.genes.get_by_id(gene_id)

        # Get associated reactions
        reactions = [
            {"id": r.id, "name": r.name, "gpr": str(r.gene_reaction_rule)}
            for r in gene.reactions
        ]

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "gene_id": gene.id,
                "name": gene.name,
                "n_reactions": len(reactions),
                "reactions": reactions,
            },
            message=(
                f"Gene {gene.id}: Associated with {len(reactions)} reaction(s)."
            ),
            message_ko=(
                f"유전자 {gene.id}: {len(reactions)}개 반응과 연관됨."
            ),
        )

    def _analyze_gene_impact(self, gene_id: str) -> SkillResult:
        """Analyze the metabolic impact of a gene knockout."""
        check = self._check_model()
        if check:
            return check

        # First, do the knockout to get growth effect
        ko_result = self._knockout_gene(gene_id)
        if not ko_result.success:
            return ko_result

        # Get gene info
        info_result = self._get_gene_info(gene_id)
        if not info_result.success:
            return info_result

        # Combine results
        ko_data = ko_result.data
        info_data = info_result.data

        impact_level = "High" if ko_data["is_essential"] else (
            "Medium" if ko_data["growth_ratio"] < 0.5 else "Low"
        )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                **ko_data,
                **info_data,
                "impact_level": impact_level,
            },
            message=(
                f"Gene {gene_id} impact analysis: {impact_level} impact. "
                f"Growth ratio: {round(ko_data['growth_ratio'] * 100, 1)}%. "
                f"Affects {len(info_data['reactions'])} reaction(s)."
            ),
            message_ko=(
                f"유전자 {gene_id} 영향 분석: {impact_level} 영향. "
                f"성장률 비율: {round(ko_data['growth_ratio'] * 100, 1)}%. "
                f"{len(info_data['reactions'])}개 반응에 영향."
            ),
        )
