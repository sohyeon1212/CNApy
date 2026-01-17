"""Shared pytest fixtures for agent tests."""

import pytest
from unittest.mock import MagicMock, patch
import cobra

from cnapy.appdata import AppData


class MockProject:
    """Mock project for testing."""

    def __init__(self, model=None):
        self.cobra_py_model = model
        self.scen_values = {}
        self.comp_values = {}
        self.comp_values_type = {}
        self.clipboard = {}
        self.scenario_modified = False
        self.solution = None  # Last optimization solution

    def load_scenario_into_model(self, model=None):
        """Mock method to load scenario into model."""
        target_model = model if model is not None else self.cobra_py_model
        if target_model is None:
            return
        # Apply scen_values to model bounds
        for rxn_id, (lb, ub) in self.scen_values.items():
            if rxn_id in target_model.reactions:
                rxn = target_model.reactions.get_by_id(rxn_id)
                rxn.bounds = (lb, ub)


class MockAppData:
    """Mock AppData for testing."""

    def __init__(self, model=None):
        self.project = MockProject(model)

    def scen_values_set(self, rxn_id, bounds):
        """Set scenario values for a reaction."""
        self.project.scen_values[rxn_id] = bounds
        self.project.scenario_modified = True

    def scen_values_clear(self):
        """Clear all scenario values."""
        self.project.scen_values.clear()
        self.project.scenario_modified = False


@pytest.fixture
def mock_appdata(simple_model):
    """Create mock AppData with a simple model."""
    return MockAppData(simple_model)


@pytest.fixture
def mock_appdata_ecoli(ecoli_core_model):
    """Create mock AppData with E. coli core model."""
    return MockAppData(ecoli_core_model)


@pytest.fixture
def simple_model():
    """Create a simple model with a few reactions for testing."""
    model = cobra.Model("simple_model")

    # Create metabolites
    A = cobra.Metabolite("A", compartment="c")
    B = cobra.Metabolite("B", compartment="c")
    C = cobra.Metabolite("C", compartment="c")

    # Create reactions
    r1 = cobra.Reaction("R1")
    r1.name = "Reaction 1"
    r1.add_metabolites({A: -1, B: 1})
    r1.bounds = (-1000, 1000)

    r2 = cobra.Reaction("R2")
    r2.name = "Reaction 2"
    r2.add_metabolites({B: -1, C: 1})
    r2.bounds = (-1000, 1000)

    r3 = cobra.Reaction("EX_A")
    r3.name = "A exchange"
    r3.add_metabolites({A: -1})
    r3.bounds = (-10, 1000)

    r4 = cobra.Reaction("EX_C")
    r4.name = "C exchange"
    r4.add_metabolites({C: -1})
    r4.bounds = (0, 1000)

    # Add gene associations
    r1.gene_reaction_rule = "gene1"
    r2.gene_reaction_rule = "gene2"

    model.add_reactions([r1, r2, r3, r4])
    model.objective = "R2"

    return model


@pytest.fixture
def ecoli_core_model():
    """Load E. coli core model for testing."""
    try:
        model = cobra.io.load_model("textbook")
        return model
    except Exception:
        pytest.skip("E. coli core model not available")


@pytest.fixture
def agent_context(mock_appdata):
    """Create AgentContext for testing."""
    from cnapy.agents.base_agent import AgentContext

    return AgentContext(
        appdata=mock_appdata,
        main_window=None,
    )


@pytest.fixture
def agent_context_ecoli(mock_appdata_ecoli):
    """Create AgentContext with E. coli model for testing."""
    from cnapy.agents.base_agent import AgentContext

    return AgentContext(
        appdata=mock_appdata_ecoli,
        main_window=None,
    )


@pytest.fixture
def mock_llm_config():
    """Create mock LLM config."""
    mock = MagicMock()
    mock.default_provider = "gemini"
    mock.gemini_api_key = "test_key"
    mock.openai_api_key = ""
    mock.anthropic_api_key = ""
    mock.use_cache = False
    mock.cache_dir = "/tmp/test_cache"
    mock.cache_expiry_days = 30
    mock.default_model_gemini = "gemini-pro"
    mock.default_model_openai = "gpt-4"
    mock.default_model_anthropic = "claude-3"
    return mock
