"""Tests for MOMA (Minimization of Metabolic Adjustment) module."""

import pytest

from cnapy.moma import has_milp_solver, linear_moma, room


class TestLinearMoma:
    """Tests for linear_moma function."""

    def test_basic_moma(self, ecoli_core_model, wildtype_fluxes):
        """Test basic linear MOMA with E. coli core model."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = linear_moma(model, wildtype_fluxes)

        assert solution is not None
        assert hasattr(solution, "status")
        assert hasattr(solution, "objective_value")

    def test_moma_optimal_status(self, ecoli_core_model, wildtype_fluxes):
        """Test that MOMA returns optimal solution."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = linear_moma(model, wildtype_fluxes)

        assert solution.status == "optimal"

    def test_moma_with_knockout(self, ecoli_core_model, wildtype_fluxes):
        """Test MOMA with a reaction knockout."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            # Find a non-essential reaction to knock out
            try:
                rxn = model.reactions.get_by_id("PFK")
                rxn.bounds = (0, 0)
            except KeyError:
                pytest.skip("PFK reaction not found in model")

            solution = linear_moma(model, wildtype_fluxes)

        assert solution is not None
        assert solution.status in ["optimal", "infeasible"]

    def test_moma_missing_reactions(self, ecoli_core_model):
        """Test MOMA with reference fluxes for missing reactions."""
        # Reference fluxes with reactions not in model
        reference_fluxes = {"MISSING_RXN_1": 5.0, "MISSING_RXN_2": 10.0}

        with ecoli_core_model as model:
            solution = linear_moma(model, reference_fluxes)

        # Should still return a solution (missing reactions are skipped)
        assert solution is not None

    def test_moma_empty_reference(self, ecoli_core_model):
        """Test MOMA with empty reference fluxes."""
        with ecoli_core_model as model:
            solution = linear_moma(model, {})

        # Should return a solution (effectively just optimizing nothing)
        assert solution is not None

    def test_moma_simple_model(self, simple_model):
        """Test MOMA with a simple model."""
        reference_fluxes = {"R1": 5.0, "R2": 5.0, "EX_A": -5.0, "EX_C": 5.0}

        solution = linear_moma(simple_model, reference_fluxes)

        assert solution is not None
        assert solution.status == "optimal"


class TestHasMilpSolver:
    """Tests for has_milp_solver function."""

    def test_returns_tuple(self):
        """Test that has_milp_solver returns a tuple."""
        result = has_milp_solver()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_bool(self):
        """Test that first element is boolean."""
        has_milp, _ = has_milp_solver()

        assert isinstance(has_milp, bool)

    def test_second_element_is_string(self):
        """Test that second element is string."""
        _, msg = has_milp_solver()

        assert isinstance(msg, str)

    def test_known_solvers_detected(self):
        """Test that known MILP solvers are properly detected."""
        has_milp, solver_name = has_milp_solver()

        # If MILP is available, solver name should be one of the known ones
        if has_milp:
            assert solver_name in ["glpk", "cplex", "gurobi"]


class TestRoom:
    """Tests for ROOM (Regulatory On/Off Minimization) function."""

    @pytest.mark.requires_milp
    def test_basic_room(self, ecoli_core_model, wildtype_fluxes):
        """Test basic ROOM with E. coli core model."""
        has_milp, _ = has_milp_solver()
        if not has_milp:
            pytest.skip("MILP solver not available")
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = room(model, wildtype_fluxes)

        assert solution is not None
        assert hasattr(solution, "status")

    @pytest.mark.requires_milp
    def test_room_optimal_status(self, ecoli_core_model, wildtype_fluxes):
        """Test that ROOM returns optimal solution."""
        has_milp, _ = has_milp_solver()
        if not has_milp:
            pytest.skip("MILP solver not available")
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = room(model, wildtype_fluxes)

        assert solution.status == "optimal"

    def test_room_no_milp_raises_error(self, ecoli_core_model, wildtype_fluxes, monkeypatch):
        """Test that ROOM raises error when no MILP solver is available."""

        def mock_has_milp_solver():
            return (False, "No MILP solver available")

        monkeypatch.setattr("cnapy.moma.has_milp_solver", mock_has_milp_solver)

        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            with pytest.raises(RuntimeError, match="ROOM requires a MILP solver"):
                room(model, wildtype_fluxes)

    @pytest.mark.requires_milp
    def test_room_custom_delta(self, ecoli_core_model, wildtype_fluxes):
        """Test ROOM with custom delta parameter."""
        has_milp, _ = has_milp_solver()
        if not has_milp:
            pytest.skip("MILP solver not available")
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = room(model, wildtype_fluxes, delta=0.1)

        assert solution is not None

    @pytest.mark.requires_milp
    def test_room_custom_epsilon(self, ecoli_core_model, wildtype_fluxes):
        """Test ROOM with custom epsilon parameter."""
        has_milp, _ = has_milp_solver()
        if not has_milp:
            pytest.skip("MILP solver not available")
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            solution = room(model, wildtype_fluxes, epsilon=0.01)

        assert solution is not None
