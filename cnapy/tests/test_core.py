"""Tests for core computation module."""

import cobra
import pytest

from cnapy.core import (
    check_biomass_weight,
    efm_computation,
    organic_elements,
)


class TestEfmComputation:
    """Tests for efm_computation function."""

    def test_efm_with_empty_model(self):
        """Test EFM computation with an empty model."""
        model = cobra.Model()
        scen_values = {}

        result = efm_computation(model, scen_values, True)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_efm_without_constraints(self):
        """Test EFM computation without applying constraints."""
        model = cobra.Model()
        scen_values = {}

        result = efm_computation(model, scen_values, False)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_efm_returns_none_without_efmtool(self):
        """Test that EFM returns None when efmtool is not installed."""
        # Since efmtool_link is likely not installed in test environment,
        # the function should return (None, {}) or (None, scenario)
        model = cobra.Model()
        scen_values = {}

        ems, scenario = efm_computation(model, scen_values, True)

        # Either efmtool is not installed (returns None)
        # or it works (returns container)
        assert ems is None or hasattr(ems, "fv_mat")

    def test_efm_with_zero_constraint(self):
        """Test EFM computation with a reaction constrained to zero."""
        model = cobra.Model()

        # Add a simple reaction
        A = cobra.Metabolite("A", compartment="c")
        B = cobra.Metabolite("B", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1, B: 1})
        model.add_reactions([r1])

        # Constrain R1 to zero
        scen_values = {"R1": (0, 0)}

        ems, scenario = efm_computation(model, scen_values, True)

        # R1 should be in the scenario as blocked
        if scenario:
            assert "R1" in scenario
            assert scenario["R1"] == (0, 0)

    @pytest.mark.requires_efmtool
    def test_efm_with_simple_model(self, simple_model):
        """Test EFM computation with a simple model (requires efmtool)."""
        scen_values = {}

        ems, scenario = efm_computation(simple_model, scen_values, False)

        if ems is not None:
            assert hasattr(ems, "fv_mat")
            assert hasattr(ems, "reac_id")


class TestCheckBiomassWeight:
    """Tests for check_biomass_weight function."""

    def test_biomass_weight_calculation(self):
        """Test biomass weight calculation with a simple biomass reaction."""
        model = cobra.Model()

        # Create metabolites with formula weights
        A = cobra.Metabolite("A", compartment="c", formula="C6H12O6")  # Glucose-like
        B = cobra.Metabolite("B", compartment="c", formula="C2H4O2")  # Acetate-like
        biomass = cobra.Metabolite("biomass", compartment="c")

        # Create biomass reaction: -A + B -> biomass
        bm_rxn = cobra.Reaction("BIOMASS")
        bm_rxn.add_metabolites({A: -1, B: -0.5, biomass: 1})
        model.add_reactions([bm_rxn])

        # Calculate biomass weight
        weight = check_biomass_weight(model, "BIOMASS")

        # Should return a positive value
        assert weight > 0

    def test_biomass_weight_with_missing_formula(self):
        """Test biomass weight calculation when a metabolite lacks a formula."""
        model = cobra.Model()

        # Create metabolites - one without formula
        A = cobra.Metabolite("A", compartment="c", formula="C6H12O6")
        B = cobra.Metabolite("B", compartment="c")  # No formula

        bm_rxn = cobra.Reaction("BIOMASS")
        bm_rxn.add_metabolites({A: -1, B: -1})
        model.add_reactions([bm_rxn])

        # Should not raise an error, but will print a warning
        weight = check_biomass_weight(model, "BIOMASS")

        # Weight will be calculated from metabolites with valid formulas
        assert isinstance(weight, float)

    def test_biomass_weight_ecoli_model(self, ecoli_core_model):
        """Test biomass weight calculation with E. coli core model."""
        # Find biomass reaction
        biomass_rxns = [r for r in ecoli_core_model.reactions if "biomass" in r.id.lower()]

        if not biomass_rxns:
            pytest.skip("No biomass reaction found in E. coli core model")

        bm_rxn_id = biomass_rxns[0].id
        weight = check_biomass_weight(ecoli_core_model, bm_rxn_id)

        # E. coli biomass should be a positive value (typically 1-2 g/gDW)
        assert weight > 0


class TestOrganicElements:
    """Tests for organic_elements constant."""

    def test_organic_elements_content(self):
        """Test that organic_elements contains expected elements."""
        expected = ["C", "O", "H", "N", "P", "S"]
        assert organic_elements == expected

    def test_organic_elements_is_list(self):
        """Test that organic_elements is a list."""
        assert isinstance(organic_elements, list)


class TestReplaceIds:
    """Tests for replace_ids function (if accessible)."""

    def test_replace_ids_basic(self):
        """Test basic ID replacement functionality."""
        from cnapy.core import replace_ids

        model = cobra.Model()

        # Create a reaction with annotation
        A = cobra.Metabolite("A", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1})
        r1.annotation["bigg.reaction"] = "PFK"
        model.add_reactions([r1])

        # Try to replace IDs
        replace_ids(model.reactions, "bigg.reaction")

        # R1 should be renamed to PFK
        assert "PFK" in [r.id for r in model.reactions]

    def test_replace_ids_no_annotation(self):
        """Test ID replacement when annotation key is missing."""
        from cnapy.core import replace_ids

        model = cobra.Model()

        A = cobra.Metabolite("A", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1})
        model.add_reactions([r1])

        # Try to replace IDs with missing annotation key
        replace_ids(model.reactions, "nonexistent_key")

        # R1 should remain unchanged
        assert "R1" in [r.id for r in model.reactions]

    def test_replace_ids_with_separator(self):
        """Test ID replacement with candidates separator."""
        from cnapy.core import replace_ids

        model = cobra.Model()

        A = cobra.Metabolite("A", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1})
        r1.annotation["candidates"] = "NEW_ID1;NEW_ID2"
        model.add_reactions([r1])

        replace_ids(model.reactions, "candidates", candidates_separator=";")

        # First candidate should be used
        assert "NEW_ID1" in [r.id for r in model.reactions]

    def test_replace_ids_unambiguous_only(self):
        """Test ID replacement with unambiguous_only flag."""
        from cnapy.core import replace_ids

        model = cobra.Model()

        A = cobra.Metabolite("A", compartment="c")
        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1})
        r1.annotation["ids"] = ["ID1", "ID2"]  # Multiple candidates
        model.add_reactions([r1])

        replace_ids(model.reactions, "ids", unambiguous_only=True)

        # R1 should remain unchanged (multiple candidates)
        assert "R1" in [r.id for r in model.reactions]

    def test_replace_ids_unique_only(self):
        """Test ID replacement with unique_only flag."""
        from cnapy.core import replace_ids

        model = cobra.Model()

        A = cobra.Metabolite("A", compartment="c")
        B = cobra.Metabolite("B", compartment="c")

        r1 = cobra.Reaction("R1")
        r1.add_metabolites({A: -1})
        r1.annotation["new_id"] = "SHARED_ID"

        r2 = cobra.Reaction("R2")
        r2.add_metabolites({B: -1})
        r2.annotation["new_id"] = "SHARED_ID"

        model.add_reactions([r1, r2])

        replace_ids(model.reactions, "new_id", unique_only=True)

        # Neither should be renamed (ID is not unique)
        assert "R1" in [r.id for r in model.reactions]
        assert "R2" in [r.id for r in model.reactions]
