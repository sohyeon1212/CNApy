"""Predefined Scenarios and Conditions for CNApy Multi-Agent System

This module defines standard culture conditions, carbon sources, nitrogen sources,
and common experimental scenarios used in metabolic modeling.

These definitions are used by the ScenarioManagerAgent to apply common
experimental conditions quickly.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CultureCondition:
    """Definition of a culture condition.

    Attributes:
        name: Condition name (internal identifier)
        display_name: Human-readable name
        display_name_ko: Korean display name
        description: Description of the condition
        description_ko: Korean description
        bounds: Dictionary of reaction_id -> (lower_bound, upper_bound)
    """

    name: str
    display_name: str
    display_name_ko: str
    description: str
    description_ko: str
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class CarbonSource:
    """Definition of a carbon source.

    Attributes:
        name: Carbon source name (internal identifier)
        display_name: Human-readable name
        display_name_ko: Korean display name
        exchange_reaction: Exchange reaction ID
        default_uptake: Default uptake rate (negative for consumption)
        alternatives: Alternative exchange reaction IDs for different models
    """

    name: str
    display_name: str
    display_name_ko: str
    exchange_reaction: str
    default_uptake: float = -10.0
    alternatives: list[str] = field(default_factory=list)


@dataclass
class NitrogenSource:
    """Definition of a nitrogen source.

    Attributes:
        name: Nitrogen source name (internal identifier)
        display_name: Human-readable name
        display_name_ko: Korean display name
        exchange_reaction: Exchange reaction ID
        default_uptake: Default uptake rate (negative for consumption)
        alternatives: Alternative exchange reaction IDs for different models
    """

    name: str
    display_name: str
    display_name_ko: str
    exchange_reaction: str
    default_uptake: float = -10.0
    alternatives: list[str] = field(default_factory=list)


@dataclass
class PredefinedWorkflow:
    """Definition of a predefined analysis workflow.

    Attributes:
        name: Workflow name (internal identifier)
        display_name: Human-readable name
        display_name_ko: Korean display name
        description: Workflow description
        description_ko: Korean description
        steps: List of workflow steps (agent, skill, params)
    """

    name: str
    display_name: str
    display_name_ko: str
    description: str
    description_ko: str
    steps: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# CULTURE CONDITIONS
# =============================================================================

CULTURE_CONDITIONS: dict[str, CultureCondition] = {
    "aerobic": CultureCondition(
        name="aerobic",
        display_name="Aerobic",
        display_name_ko="호기성",
        description="Aerobic condition with oxygen uptake enabled",
        description_ko="산소 섭취가 가능한 호기성 조건",
        bounds={
            "EX_o2_e": (-20.0, 1000.0),  # Standard E. coli model
        },
    ),
    "anaerobic": CultureCondition(
        name="anaerobic",
        display_name="Anaerobic",
        display_name_ko="혐기성",
        description="Anaerobic condition with no oxygen uptake",
        description_ko="산소 섭취가 없는 혐기성 조건",
        bounds={
            "EX_o2_e": (0.0, 0.0),
        },
    ),
    "microaerobic": CultureCondition(
        name="microaerobic",
        display_name="Microaerobic",
        display_name_ko="미호기성",
        description="Microaerobic condition with limited oxygen uptake",
        description_ko="제한된 산소 섭취를 가진 미호기성 조건",
        bounds={
            "EX_o2_e": (-2.0, 0.0),
        },
    ),
    "minimal_media": CultureCondition(
        name="minimal_media",
        display_name="Minimal Media (M9)",
        display_name_ko="최소 배지 (M9)",
        description="Minimal M9 media conditions",
        description_ko="최소 M9 배지 조건",
        bounds={
            "EX_o2_e": (-20.0, 1000.0),
            "EX_glc__D_e": (-10.0, 0.0),
            "EX_nh4_e": (-10.0, 1000.0),
            "EX_pi_e": (-10.0, 1000.0),
            "EX_so4_e": (-10.0, 1000.0),
            "EX_k_e": (-10.0, 1000.0),
            "EX_mg2_e": (-10.0, 1000.0),
            "EX_ca2_e": (-10.0, 1000.0),
            "EX_fe2_e": (-10.0, 1000.0),
            "EX_cl_e": (-10.0, 1000.0),
            "EX_na1_e": (-10.0, 1000.0),
        },
    ),
    "rich_media": CultureCondition(
        name="rich_media",
        display_name="Rich Media (LB)",
        display_name_ko="풍부 배지 (LB)",
        description="Rich LB media conditions with amino acids",
        description_ko="아미노산이 포함된 풍부한 LB 배지 조건",
        bounds={
            "EX_o2_e": (-20.0, 1000.0),
            "EX_glc__D_e": (-10.0, 0.0),
        },
    ),
}

# Alternative reaction IDs for different models
OXYGEN_EXCHANGE_ALTERNATIVES = [
    "EX_o2_e",
    "EX_o2(e)",
    "R_EX_o2_e",
    "EX_o2_e_",
    "O2t",
]


# =============================================================================
# CARBON SOURCES
# =============================================================================

CARBON_SOURCES: dict[str, CarbonSource] = {
    "glucose": CarbonSource(
        name="glucose",
        display_name="Glucose",
        display_name_ko="포도당",
        exchange_reaction="EX_glc__D_e",
        default_uptake=-10.0,
        alternatives=["EX_glc_D_e", "EX_glc(e)", "R_EX_glc__D_e", "GLCt1"],
    ),
    "xylose": CarbonSource(
        name="xylose",
        display_name="Xylose",
        display_name_ko="자일로스",
        exchange_reaction="EX_xyl__D_e",
        default_uptake=-10.0,
        alternatives=["EX_xyl_D_e", "EX_xyl(e)", "R_EX_xyl__D_e"],
    ),
    "glycerol": CarbonSource(
        name="glycerol",
        display_name="Glycerol",
        display_name_ko="글리세롤",
        exchange_reaction="EX_glyc_e",
        default_uptake=-10.0,
        alternatives=["EX_glyc(e)", "R_EX_glyc_e"],
    ),
    "acetate": CarbonSource(
        name="acetate",
        display_name="Acetate",
        display_name_ko="아세테이트",
        exchange_reaction="EX_ac_e",
        default_uptake=-10.0,
        alternatives=["EX_ac(e)", "R_EX_ac_e"],
    ),
    "lactate": CarbonSource(
        name="lactate",
        display_name="Lactate",
        display_name_ko="락테이트",
        exchange_reaction="EX_lac__D_e",
        default_uptake=-10.0,
        alternatives=["EX_lac_D_e", "EX_lac(e)", "R_EX_lac__D_e"],
    ),
    "succinate": CarbonSource(
        name="succinate",
        display_name="Succinate",
        display_name_ko="숙시네이트",
        exchange_reaction="EX_succ_e",
        default_uptake=-10.0,
        alternatives=["EX_succ(e)", "R_EX_succ_e"],
    ),
    "fructose": CarbonSource(
        name="fructose",
        display_name="Fructose",
        display_name_ko="과당",
        exchange_reaction="EX_fru_e",
        default_uptake=-10.0,
        alternatives=["EX_fru(e)", "R_EX_fru_e"],
    ),
    "galactose": CarbonSource(
        name="galactose",
        display_name="Galactose",
        display_name_ko="갈락토스",
        exchange_reaction="EX_gal_e",
        default_uptake=-10.0,
        alternatives=["EX_gal(e)", "R_EX_gal_e"],
    ),
}


# =============================================================================
# NITROGEN SOURCES
# =============================================================================

NITROGEN_SOURCES: dict[str, NitrogenSource] = {
    "ammonium": NitrogenSource(
        name="ammonium",
        display_name="Ammonium (NH4+)",
        display_name_ko="암모늄 (NH4+)",
        exchange_reaction="EX_nh4_e",
        default_uptake=-10.0,
        alternatives=["EX_nh4(e)", "R_EX_nh4_e"],
    ),
    "nitrate": NitrogenSource(
        name="nitrate",
        display_name="Nitrate (NO3-)",
        display_name_ko="질산염 (NO3-)",
        exchange_reaction="EX_no3_e",
        default_uptake=-10.0,
        alternatives=["EX_no3(e)", "R_EX_no3_e"],
    ),
    "glutamate": NitrogenSource(
        name="glutamate",
        display_name="Glutamate",
        display_name_ko="글루타메이트",
        exchange_reaction="EX_glu__L_e",
        default_uptake=-10.0,
        alternatives=["EX_glu_L_e", "EX_glu(e)", "R_EX_glu__L_e"],
    ),
    "glutamine": NitrogenSource(
        name="glutamine",
        display_name="Glutamine",
        display_name_ko="글루타민",
        exchange_reaction="EX_gln__L_e",
        default_uptake=-10.0,
        alternatives=["EX_gln_L_e", "EX_gln(e)", "R_EX_gln__L_e"],
    ),
}


# =============================================================================
# COMMON OBJECTIVE FUNCTIONS
# =============================================================================

COMMON_OBJECTIVES: dict[str, dict[str, Any]] = {
    "biomass": {
        "display_name": "Biomass (Growth)",
        "display_name_ko": "바이오매스 (성장)",
        "reaction_patterns": [
            "BIOMASS_Ec_iJO1366_core_53p95M",
            "BIOMASS_Ec_iML1515_core_75p37M",
            "BIOMASS_",
            "biomass",
            "Biomass",
            "GROWTH",
            "growth",
        ],
        "direction": "max",
    },
    "atp_maintenance": {
        "display_name": "ATP Maintenance",
        "display_name_ko": "ATP 유지",
        "reaction_patterns": ["ATPM", "ATPS4r"],
        "direction": "max",
    },
}


# =============================================================================
# PREDEFINED WORKFLOWS
# =============================================================================

PREDEFINED_WORKFLOWS: dict[str, PredefinedWorkflow] = {
    "fba_aerobic": PredefinedWorkflow(
        name="fba_aerobic",
        display_name="FBA under Aerobic Conditions",
        display_name_ko="호기성 조건에서 FBA",
        description="Perform FBA with aerobic conditions",
        description_ko="호기성 조건에서 FBA 수행",
        steps=[
            {"agent": "scenario", "skill": "apply_condition", "params": {"condition_name": "aerobic"}},
            {"agent": "flux_analysis", "skill": "perform_fba", "params": {}},
        ],
    ),
    "fba_anaerobic": PredefinedWorkflow(
        name="fba_anaerobic",
        display_name="FBA under Anaerobic Conditions",
        display_name_ko="혐기성 조건에서 FBA",
        description="Perform FBA with anaerobic conditions",
        description_ko="혐기성 조건에서 FBA 수행",
        steps=[
            {"agent": "scenario", "skill": "apply_condition", "params": {"condition_name": "anaerobic"}},
            {"agent": "flux_analysis", "skill": "perform_fba", "params": {}},
        ],
    ),
    "essential_genes": PredefinedWorkflow(
        name="essential_genes",
        display_name="Find Essential Genes",
        display_name_ko="필수 유전자 탐색",
        description="Find essential genes under current conditions",
        description_ko="현재 조건에서 필수 유전자 탐색",
        steps=[
            {"agent": "gene_analysis", "skill": "find_essential_genes", "params": {"threshold": 0.01}},
        ],
    ),
    "gene_knockout_fba": PredefinedWorkflow(
        name="gene_knockout_fba",
        display_name="Gene Knockout + FBA",
        display_name_ko="유전자 녹아웃 + FBA",
        description="Knockout a gene and perform FBA",
        description_ko="유전자를 녹아웃하고 FBA 수행",
        steps=[
            {"agent": "gene_analysis", "skill": "knockout_gene", "params": {"gene_id": ""}},  # gene_id to be filled
            {"agent": "flux_analysis", "skill": "perform_fba", "params": {}},
        ],
    ),
    "fva_analysis": PredefinedWorkflow(
        name="fva_analysis",
        display_name="FVA Analysis",
        display_name_ko="FVA 분석",
        description="Perform Flux Variability Analysis",
        description_ko="플럭스 변동성 분석 수행",
        steps=[
            {"agent": "flux_analysis", "skill": "perform_fva", "params": {"fraction_of_optimum": 0.9}},
        ],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_exchange_reaction(model, reaction_patterns: list[str]) -> str | None:
    """Find an exchange reaction in the model by trying multiple patterns.

    Args:
        model: COBRA model
        reaction_patterns: List of reaction ID patterns to try

    Returns:
        First matching reaction ID or None
    """
    reaction_ids = {r.id for r in model.reactions}

    for pattern in reaction_patterns:
        if pattern in reaction_ids:
            return pattern

    return None


def get_culture_condition(name: str) -> CultureCondition | None:
    """Get a culture condition by name.

    Args:
        name: Condition name (case-insensitive)

    Returns:
        CultureCondition or None if not found
    """
    return CULTURE_CONDITIONS.get(name.lower())


def get_carbon_source(name: str) -> CarbonSource | None:
    """Get a carbon source by name.

    Args:
        name: Carbon source name (case-insensitive)

    Returns:
        CarbonSource or None if not found
    """
    return CARBON_SOURCES.get(name.lower())


def get_nitrogen_source(name: str) -> NitrogenSource | None:
    """Get a nitrogen source by name.

    Args:
        name: Nitrogen source name (case-insensitive)

    Returns:
        NitrogenSource or None if not found
    """
    return NITROGEN_SOURCES.get(name.lower())


def get_workflow(name: str) -> PredefinedWorkflow | None:
    """Get a predefined workflow by name.

    Args:
        name: Workflow name (case-insensitive)

    Returns:
        PredefinedWorkflow or None if not found
    """
    return PREDEFINED_WORKFLOWS.get(name.lower())


def list_conditions() -> list[tuple[str, str, str]]:
    """List all available culture conditions.

    Returns:
        List of (name, display_name, display_name_ko) tuples
    """
    return [(c.name, c.display_name, c.display_name_ko) for c in CULTURE_CONDITIONS.values()]


def list_carbon_sources() -> list[tuple[str, str, str]]:
    """List all available carbon sources.

    Returns:
        List of (name, display_name, display_name_ko) tuples
    """
    return [(c.name, c.display_name, c.display_name_ko) for c in CARBON_SOURCES.values()]


def list_nitrogen_sources() -> list[tuple[str, str, str]]:
    """List all available nitrogen sources.

    Returns:
        List of (name, display_name, display_name_ko) tuples
    """
    return [(n.name, n.display_name, n.display_name_ko) for n in NITROGEN_SOURCES.values()]


def list_workflows() -> list[tuple[str, str, str]]:
    """List all available predefined workflows.

    Returns:
        List of (name, display_name, display_name_ko) tuples
    """
    return [(w.name, w.display_name, w.display_name_ko) for w in PREDEFINED_WORKFLOWS.values()]
