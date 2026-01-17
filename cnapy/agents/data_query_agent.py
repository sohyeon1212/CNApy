"""Data Query Agent for CNApy Multi-Agent System

This agent handles model information queries:
- Model statistics and information
- Reaction/metabolite/gene information
- Search functionality
- Pathway queries
- Analysis result retrieval
"""


from cnapy.agents.base_agent import (
    BaseAgent,
    Skill,
    SkillResult,
    SkillStatus,
)


class DataQueryAgent(BaseAgent):
    """Agent for querying model information.

    Handles information retrieval, search, and data export.
    """

    @property
    def name(self) -> str:
        return "data_query"

    @property
    def description(self) -> str:
        return "Queries model information, searches reactions/metabolites/genes, and retrieves analysis results."

    @property
    def description_ko(self) -> str:
        return "모델 정보 조회, 반응/대사체/유전자 검색, 분석 결과 조회를 수행합니다."

    def _register_skills(self):
        """Register all data query skills."""

        # Get model info
        self.register_skill(
            Skill(
                name="get_model_info",
                description="Get basic model statistics and information",
                description_ko="모델 기본 통계 및 정보 조회",
                parameters={},
                required_params=[],
                handler=self._get_model_info,
            )
        )

        # Get reaction info
        self.register_skill(
            Skill(
                name="get_reaction_info",
                description="Get detailed information about a specific reaction",
                description_ko="특정 반응의 상세 정보 조회",
                parameters={
                    "reaction_id": {
                        "type": "string",
                        "description": "Reaction ID to query",
                    },
                },
                required_params=["reaction_id"],
                handler=self._get_reaction_info,
            )
        )

        # Get metabolite info
        self.register_skill(
            Skill(
                name="get_metabolite_info",
                description="Get detailed information about a specific metabolite",
                description_ko="특정 대사체의 상세 정보 조회",
                parameters={
                    "metabolite_id": {
                        "type": "string",
                        "description": "Metabolite ID to query",
                    },
                },
                required_params=["metabolite_id"],
                handler=self._get_metabolite_info,
            )
        )

        # Get gene info
        self.register_skill(
            Skill(
                name="get_gene_info",
                description="Get detailed information about a specific gene",
                description_ko="특정 유전자의 상세 정보 조회",
                parameters={
                    "gene_id": {
                        "type": "string",
                        "description": "Gene ID to query",
                    },
                },
                required_params=["gene_id"],
                handler=self._get_gene_info,
            )
        )

        # Search reactions
        self.register_skill(
            Skill(
                name="search_reactions",
                description="Search for reactions by ID, name, or equation",
                description_ko="ID, 이름 또는 반응식으로 반응 검색",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "search_in": {
                        "type": "string",
                        "description": "Where to search (id, name, equation, all)",
                        "enum": ["id", "name", "equation", "all"],
                        "default": "all",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                },
                required_params=["query"],
                handler=self._search_reactions,
            )
        )

        # Search metabolites
        self.register_skill(
            Skill(
                name="search_metabolites",
                description="Search for metabolites by ID or name",
                description_ko="ID 또는 이름으로 대사체 검색",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                },
                required_params=["query"],
                handler=self._search_metabolites,
            )
        )

        # Search genes
        self.register_skill(
            Skill(
                name="search_genes",
                description="Search for genes by ID or name",
                description_ko="ID 또는 이름으로 유전자 검색",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                },
                required_params=["query"],
                handler=self._search_genes,
            )
        )

        # Get exchange reactions
        self.register_skill(
            Skill(
                name="get_exchange_reactions",
                description="Get all exchange reactions in the model",
                description_ko="모델의 모든 교환 반응 조회",
                parameters={},
                required_params=[],
                handler=self._get_exchange_reactions,
            )
        )

        # Get objective
        self.register_skill(
            Skill(
                name="get_objective",
                description="Get the current objective function",
                description_ko="현재 목적함수 조회",
                parameters={},
                required_params=[],
                handler=self._get_objective,
            )
        )

        # List compartments
        self.register_skill(
            Skill(
                name="list_compartments",
                description="List all compartments in the model",
                description_ko="모델의 모든 구획 목록",
                parameters={},
                required_params=[],
                handler=self._list_compartments,
            )
        )

        # Get analysis results
        self.register_skill(
            Skill(
                name="get_analysis_results",
                description="Get results from recent analysis",
                description_ko="최근 분석 결과 조회",
                parameters={
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis (fba, fva, essential_genes, etc.)",
                        "default": "latest",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 10,
                    },
                },
                required_params=[],
                handler=self._get_analysis_results,
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

    def _get_model_info(self) -> SkillResult:
        """Get model statistics and information."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        # Count exchange reactions
        exchange_rxns = [r for r in model.reactions if r.id.startswith("EX_")]

        # Get objective
        obj_str = str(model.objective.expression) if model.objective else "None"

        info = {
            "model_id": model.id,
            "model_name": model.name,
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
            "n_exchange_reactions": len(exchange_rxns),
            "n_compartments": len(model.compartments),
            "compartments": list(model.compartments.keys()),
            "objective": obj_str[:100] if obj_str else "None",
        }

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data=info,
            message=(
                f"Model: {model.name or model.id}\n"
                f"• Reactions: {info['n_reactions']}\n"
                f"• Metabolites: {info['n_metabolites']}\n"
                f"• Genes: {info['n_genes']}\n"
                f"• Exchange reactions: {info['n_exchange_reactions']}\n"
                f"• Compartments: {', '.join(info['compartments'])}"
            ),
            message_ko=(
                f"모델: {model.name or model.id}\n"
                f"• 반응: {info['n_reactions']}개\n"
                f"• 대사체: {info['n_metabolites']}개\n"
                f"• 유전자: {info['n_genes']}개\n"
                f"• 교환 반응: {info['n_exchange_reactions']}개\n"
                f"• 구획: {', '.join(info['compartments'])}"
            ),
        )

    def _get_reaction_info(self, reaction_id: str) -> SkillResult:
        """Get detailed information about a reaction."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        if reaction_id not in [r.id for r in model.reactions]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Reaction not found: {reaction_id}",
                message=f"Reaction '{reaction_id}' not found in the model.",
                message_ko=f"반응 '{reaction_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        rxn = model.reactions.get_by_id(reaction_id)

        # Get current flux if available
        flux = None
        if self.context.comp_values and reaction_id in self.context.comp_values:
            flux = self.context.comp_values[reaction_id]

        info = {
            "id": rxn.id,
            "name": rxn.name,
            "reaction": rxn.reaction,
            "lower_bound": rxn.lower_bound,
            "upper_bound": rxn.upper_bound,
            "gene_reaction_rule": rxn.gene_reaction_rule,
            "subsystem": rxn.subsystem,
            "reversibility": rxn.reversibility,
            "n_genes": len(rxn.genes),
            "genes": [g.id for g in rxn.genes],
            "n_metabolites": len(rxn.metabolites),
            "current_flux": flux,
        }

        message = (
            f"Reaction {rxn.id}: {rxn.name or 'N/A'}\n"
            f"• Equation: {rxn.reaction}\n"
            f"• Bounds: [{rxn.lower_bound}, {rxn.upper_bound}]\n"
            f"• GPR: {rxn.gene_reaction_rule or 'None'}\n"
            f"• Subsystem: {rxn.subsystem or 'None'}"
        )

        if flux:
            message += f"\n• Current flux: [{flux[0]}, {flux[1]}]"

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data=info,
            message=message,
            message_ko=message,
        )

    def _get_metabolite_info(self, metabolite_id: str) -> SkillResult:
        """Get detailed information about a metabolite."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        if metabolite_id not in [m.id for m in model.metabolites]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Metabolite not found: {metabolite_id}",
                message=f"Metabolite '{metabolite_id}' not found in the model.",
                message_ko=f"대사체 '{metabolite_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        met = model.metabolites.get_by_id(metabolite_id)

        # Get producing and consuming reactions
        producing = []
        consuming = []
        for rxn in met.reactions:
            coeff = rxn.metabolites[met]
            if coeff > 0:
                producing.append(rxn.id)
            elif coeff < 0:
                consuming.append(rxn.id)

        info = {
            "id": met.id,
            "name": met.name,
            "formula": met.formula,
            "charge": met.charge,
            "compartment": met.compartment,
            "n_reactions": len(met.reactions),
            "producing_reactions": producing,
            "consuming_reactions": consuming,
        }

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data=info,
            message=(
                f"Metabolite {met.id}: {met.name or 'N/A'}\n"
                f"• Formula: {met.formula or 'N/A'}\n"
                f"• Charge: {met.charge}\n"
                f"• Compartment: {met.compartment}\n"
                f"• Reactions: {len(met.reactions)}"
            ),
            message_ko=(
                f"대사체 {met.id}: {met.name or 'N/A'}\n"
                f"• 화학식: {met.formula or 'N/A'}\n"
                f"• 전하: {met.charge}\n"
                f"• 구획: {met.compartment}\n"
                f"• 관련 반응: {len(met.reactions)}개"
            ),
        )

    def _get_gene_info(self, gene_id: str) -> SkillResult:
        """Get detailed information about a gene."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        if gene_id not in [g.id for g in model.genes]:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error=f"Gene not found: {gene_id}",
                message=f"Gene '{gene_id}' not found in the model.",
                message_ko=f"유전자 '{gene_id}'을(를) 모델에서 찾을 수 없습니다.",
            )

        gene = model.genes.get_by_id(gene_id)

        reactions = [{"id": r.id, "name": r.name} for r in gene.reactions]

        info = {
            "id": gene.id,
            "name": gene.name,
            "n_reactions": len(reactions),
            "reactions": reactions,
        }

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data=info,
            message=(f"Gene {gene.id}: {gene.name or 'N/A'}\n" f"• Associated reactions: {len(reactions)}"),
            message_ko=(f"유전자 {gene.id}: {gene.name or 'N/A'}\n" f"• 연관 반응: {len(reactions)}개"),
        )

    def _search_reactions(
        self,
        query: str,
        search_in: str = "all",
        max_results: int = 20,
    ) -> SkillResult:
        """Search for reactions."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model
        query_lower = query.lower()
        results = []

        for rxn in model.reactions:
            match = False
            match_field = ""

            if search_in in ["id", "all"]:
                if query_lower in rxn.id.lower():
                    match = True
                    match_field = "id"

            if not match and search_in in ["name", "all"]:
                if rxn.name and query_lower in rxn.name.lower():
                    match = True
                    match_field = "name"

            if not match and search_in in ["equation", "all"]:
                if query_lower in rxn.reaction.lower():
                    match = True
                    match_field = "equation"

            if match:
                results.append(
                    {
                        "id": rxn.id,
                        "name": rxn.name,
                        "reaction": rxn.reaction[:100] + "..." if len(rxn.reaction) > 100 else rxn.reaction,
                        "match_field": match_field,
                    }
                )

            if len(results) >= max_results:
                break

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "query": query,
                "n_results": len(results),
                "results": results,
            },
            message=f"Found {len(results)} reaction(s) matching '{query}'.",
            message_ko=f"'{query}'와 일치하는 {len(results)}개 반응 발견.",
        )

    def _search_metabolites(
        self,
        query: str,
        max_results: int = 20,
    ) -> SkillResult:
        """Search for metabolites."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model
        query_lower = query.lower()
        results = []

        for met in model.metabolites:
            match = False

            if query_lower in met.id.lower():
                match = True
            elif met.name and query_lower in met.name.lower():
                match = True

            if match:
                results.append(
                    {
                        "id": met.id,
                        "name": met.name,
                        "formula": met.formula,
                        "compartment": met.compartment,
                    }
                )

            if len(results) >= max_results:
                break

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "query": query,
                "n_results": len(results),
                "results": results,
            },
            message=f"Found {len(results)} metabolite(s) matching '{query}'.",
            message_ko=f"'{query}'와 일치하는 {len(results)}개 대사체 발견.",
        )

    def _search_genes(
        self,
        query: str,
        max_results: int = 20,
    ) -> SkillResult:
        """Search for genes."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model
        query_lower = query.lower()
        results = []

        for gene in model.genes:
            match = False

            if query_lower in gene.id.lower():
                match = True
            elif gene.name and query_lower in gene.name.lower():
                match = True

            if match:
                results.append(
                    {
                        "id": gene.id,
                        "name": gene.name,
                        "n_reactions": len(gene.reactions),
                    }
                )

            if len(results) >= max_results:
                break

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "query": query,
                "n_results": len(results),
                "results": results,
            },
            message=f"Found {len(results)} gene(s) matching '{query}'.",
            message_ko=f"'{query}'와 일치하는 {len(results)}개 유전자 발견.",
        )

    def _get_exchange_reactions(self) -> SkillResult:
        """Get all exchange reactions."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        exchange_rxns = []
        for rxn in model.reactions:
            # Check if it's an exchange reaction
            if rxn.id.startswith("EX_") or len(rxn.metabolites) == 1:
                flux = None
                if self.context.comp_values and rxn.id in self.context.comp_values:
                    flux = self.context.comp_values[rxn.id]

                exchange_rxns.append(
                    {
                        "id": rxn.id,
                        "name": rxn.name,
                        "bounds": [rxn.lower_bound, rxn.upper_bound],
                        "current_flux": flux,
                    }
                )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "n_exchange": len(exchange_rxns),
                "reactions": exchange_rxns,
            },
            message=f"Found {len(exchange_rxns)} exchange reactions.",
            message_ko=f"{len(exchange_rxns)}개 교환 반응 발견.",
        )

    def _get_objective(self) -> SkillResult:
        """Get the current objective function."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        obj = model.objective
        obj_expr = str(obj.expression) if obj else "None"
        obj_dir = obj.direction if obj else "max"

        # Try to get objective value from solution
        obj_value = None
        if self.context.solution:
            obj_value = self.context.solution.objective_value

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "expression": obj_expr,
                "direction": obj_dir,
                "value": obj_value,
            },
            message=(
                f"Objective: {obj_dir}imize {obj_expr[:50]}..."
                + (f"\nCurrent value: {round(obj_value, 4)}" if obj_value else "")
            ),
            message_ko=(
                f"목적함수: {obj_dir}imize {obj_expr[:50]}..."
                + (f"\n현재 값: {round(obj_value, 4)}" if obj_value else "")
            ),
        )

    def _list_compartments(self) -> SkillResult:
        """List all compartments."""
        check = self._check_model()
        if check:
            return check

        model = self.context.model

        compartments = []
        for comp_id, comp_name in model.compartments.items():
            n_mets = sum(1 for m in model.metabolites if m.compartment == comp_id)
            compartments.append(
                {
                    "id": comp_id,
                    "name": comp_name,
                    "n_metabolites": n_mets,
                }
            )

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "n_compartments": len(compartments),
                "compartments": compartments,
            },
            message=f"Model has {len(compartments)} compartments: {', '.join(c['id'] for c in compartments)}",
            message_ko=f"모델에 {len(compartments)}개 구획: {', '.join(c['id'] for c in compartments)}",
        )

    def _get_analysis_results(
        self,
        analysis_type: str = "latest",
        top_n: int = 10,
    ) -> SkillResult:
        """Get results from recent analysis."""
        # Check for current computed values
        if not self.context.comp_values:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error="No analysis results",
                message="No analysis results available. Run an analysis first.",
                message_ko="사용 가능한 분석 결과가 없습니다. 먼저 분석을 실행하세요.",
            )

        # Get top fluxes by absolute value
        fluxes = []
        for rxn_id, (lb, ub) in self.context.comp_values.items():
            mean_flux = (lb + ub) / 2
            fluxes.append(
                {
                    "reaction_id": rxn_id,
                    "lower": lb,
                    "upper": ub,
                    "mean": mean_flux,
                    "abs_mean": abs(mean_flux),
                }
            )

        # Sort by absolute mean flux
        fluxes.sort(key=lambda x: x["abs_mean"], reverse=True)
        top_fluxes = fluxes[:top_n]

        # Get objective value if available
        obj_value = None
        if self.context.solution:
            obj_value = self.context.solution.objective_value

        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "n_total": len(fluxes),
                "top_n": top_n,
                "top_fluxes": top_fluxes,
                "objective_value": obj_value,
            },
            message=(
                f"Analysis results: {len(fluxes)} reactions with flux values."
                + (f" Objective value: {round(obj_value, 4)}" if obj_value else "")
            ),
            message_ko=(
                f"분석 결과: {len(fluxes)}개 반응에 플럭스 값."
                + (f" 목적함수 값: {round(obj_value, 4)}" if obj_value else "")
            ),
        )
