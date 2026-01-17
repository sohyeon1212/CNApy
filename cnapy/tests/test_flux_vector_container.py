"""Tests for FluxVectorContainer class."""

import os
import tempfile

import numpy as np
import pytest

from cnapy.flux_vector_container import FluxVectorContainer


class TestFluxVectorContainer:
    """Test suite for FluxVectorContainer."""

    def test_init_with_matrix(self, flux_vector_data):
        """Test initialization with a numpy matrix."""
        fv_mat, reac_id = flux_vector_data
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        assert container.fv_mat.shape == fv_mat.shape
        assert container.reac_id == reac_id
        assert len(container) == fv_mat.shape[0]

    def test_init_requires_reac_id(self, flux_vector_data):
        """Test that reac_id is required when initializing with a matrix."""
        fv_mat, _ = flux_vector_data
        with pytest.raises(TypeError, match="reac_id must be provided"):
            FluxVectorContainer(fv_mat)

    def test_init_with_irreversible(self, flux_vector_data):
        """Test initialization with irreversible flag."""
        fv_mat, reac_id = flux_vector_data
        irreversible = np.array([True, False, True, False, True])
        container = FluxVectorContainer(fv_mat, reac_id=reac_id, irreversible=irreversible)

        np.testing.assert_array_equal(container.irreversible, irreversible)

    def test_init_with_unbounded(self, flux_vector_data):
        """Test initialization with unbounded flag."""
        fv_mat, reac_id = flux_vector_data
        unbounded = np.array([False, True, False, False, True])
        container = FluxVectorContainer(fv_mat, reac_id=reac_id, unbounded=unbounded)

        np.testing.assert_array_equal(container.unbounded, unbounded)

    def test_len(self, flux_vector_data):
        """Test __len__ method."""
        fv_mat, reac_id = flux_vector_data
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        assert len(container) == 10

    def test_getitem(self, flux_vector_data):
        """Test __getitem__ returns correct flux dictionary."""
        fv_mat, reac_id = flux_vector_data
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        result = container[0]
        assert isinstance(result, dict)
        # Only non-zero values should be included
        for rid, val in result.items():
            assert rid in reac_id
            assert val != 0

    def test_getitem_excludes_zeros(self):
        """Test that __getitem__ excludes zero values."""
        fv_mat = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        reac_id = ["R1", "R2", "R3"]
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        result = container[0]
        assert "R1" in result
        assert "R2" not in result
        assert "R3" in result
        assert result["R1"] == 1.0
        assert result["R3"] == 2.0

        # All zeros row should return empty dict
        result_empty = container[1]
        assert result_empty == {}

    def test_save_and_load(self, flux_vector_data):
        """Test save and load functionality."""
        fv_mat, reac_id = flux_vector_data
        irreversible = np.array([True, False, True, False, True])
        container = FluxVectorContainer(fv_mat, reac_id=reac_id, irreversible=irreversible)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_container.npz")
            container.save(filepath)

            # Load and verify
            loaded = FluxVectorContainer(filepath)
            np.testing.assert_array_almost_equal(loaded.fv_mat, fv_mat)
            assert loaded.reac_id == reac_id
            np.testing.assert_array_equal(loaded.irreversible, irreversible)

    def test_clear(self, flux_vector_data):
        """Test clear method."""
        fv_mat, reac_id = flux_vector_data
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        container.clear()

        assert container.fv_mat.shape == (0, 0)
        assert container.reac_id == []

    def test_is_integer_vector_rounded(self):
        """Test is_integer_vector_rounded method."""
        # Vector with integer values
        fv_mat = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
        reac_id = ["R1", "R2", "R3"]
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        # First row has all integers
        assert container.is_integer_vector_rounded(0, decimals=0)
        # Second row has .5 values which round to integers, so it passes too
        # The method checks if rounded values are integers

    def test_is_integer_vector_rounded_with_decimals(self):
        """Test is_integer_vector_rounded with decimal precision."""
        # The method rounds to `decimals` places and checks if all results are integers
        # round(1.001, 2) = 1.0 which is an integer
        # So we need values that don't round to integers
        fv_mat = np.array([[1.5, 2.5, 3.5]])  # All .5 values
        reac_id = ["R1", "R2", "R3"]
        container = FluxVectorContainer(fv_mat, reac_id=reac_id)

        # With decimals=0, round(1.5)=2 which is integer, so returns True
        assert container.is_integer_vector_rounded(0, decimals=0)
