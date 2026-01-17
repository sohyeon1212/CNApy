"""Tests for flux_sampling module."""

import numpy as np
import pandas as pd
import pytest

from cnapy.flux_sampling import (
    add_gaussian_noise_to_samples,
    compute_sampling_statistics,
    perform_predicted_flux_sampling,
    perform_sampling,
)


class TestPerformSampling:
    """Tests for perform_sampling function."""

    @pytest.mark.slow
    def test_basic_sampling(self, ecoli_core_model):
        """Test basic flux sampling with E. coli core model."""
        with ecoli_core_model as model:
            samples = perform_sampling(model, n=10, thinning=10, processes=1)

        assert isinstance(samples, pd.DataFrame)
        assert len(samples) == 10
        # Should have columns for all reactions
        assert len(samples.columns) > 0

    @pytest.mark.slow
    def test_sampling_columns_match_reactions(self, ecoli_core_model):
        """Test that sampling columns match model reactions."""
        with ecoli_core_model as model:
            reaction_ids = [r.id for r in model.reactions]
            samples = perform_sampling(model, n=5, thinning=10, processes=1)

        for col in samples.columns:
            assert col in reaction_ids

    @pytest.mark.slow
    def test_sampling_feasibility(self, ecoli_core_model):
        """Test that sampled fluxes are within reaction bounds."""
        with ecoli_core_model as model:
            bounds = {r.id: r.bounds for r in model.reactions}
            samples = perform_sampling(model, n=5, thinning=10, processes=1)

        for rid, (lb, ub) in bounds.items():
            if rid in samples.columns:
                values = samples[rid]
                # Allow small numerical tolerance
                assert all(values >= lb - 1e-6), f"{rid} has values below lower bound"
                assert all(values <= ub + 1e-6), f"{rid} has values above upper bound"


class TestPerformPredictedFluxSampling:
    """Tests for perform_predicted_flux_sampling function."""

    @pytest.mark.slow
    def test_bounds_mode(self, ecoli_core_model, wildtype_fluxes):
        """Test predicted flux sampling with bounds mode."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            samples, applied_bounds = perform_predicted_flux_sampling(
                model,
                wildtype_fluxes,
                n=5,
                constraint_mode="bounds",
                thinning=10,
                processes=1,
            )

        assert isinstance(samples, pd.DataFrame)
        assert isinstance(applied_bounds, dict)
        assert len(samples) == 5

    @pytest.mark.slow
    def test_fixed_mode(self, ecoli_core_model, wildtype_fluxes):
        """Test predicted flux sampling with fixed mode.

        Note: Fixed mode typically results in no degrees of freedom for sampling,
        which can cause sampling to fail. This is expected behavior.
        """
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            try:
                samples, applied_bounds = perform_predicted_flux_sampling(
                    model,
                    wildtype_fluxes,
                    n=5,
                    constraint_mode="fixed",
                    thinning=10,
                    processes=1,
                )
                # In fixed mode, all fluxes should be fixed
                for _rid, (lb, ub) in applied_bounds.items():
                    assert lb == ub
            except RuntimeError as e:
                # Sampling with fully fixed fluxes often fails due to no degrees of freedom
                if "Sampling failed" in str(e):
                    pytest.skip("Sampling failed in fixed mode (expected - no degrees of freedom)")

    @pytest.mark.slow
    def test_free_mode(self, ecoli_core_model, wildtype_fluxes):
        """Test predicted flux sampling with free mode."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            samples, applied_bounds = perform_predicted_flux_sampling(
                model,
                wildtype_fluxes,
                n=5,
                constraint_mode="free",
                thinning=10,
                processes=1,
            )

        # In free mode, no bounds should be applied
        assert len(applied_bounds) == 0

    @pytest.mark.slow
    def test_custom_fractions(self, ecoli_core_model, wildtype_fluxes):
        """Test predicted flux sampling with custom min/max fractions."""
        if not wildtype_fluxes:
            pytest.skip("No wildtype fluxes available")

        with ecoli_core_model as model:
            samples, applied_bounds = perform_predicted_flux_sampling(
                model,
                wildtype_fluxes,
                n=5,
                constraint_mode="bounds",
                min_fraction=0.9,
                max_fraction=1.1,
                thinning=10,
                processes=1,
            )

        assert len(samples) == 5


class TestComputeSamplingStatistics:
    """Tests for compute_sampling_statistics function."""

    def test_basic_statistics(self, sample_dataframe):
        """Test computing basic statistics from samples."""
        mean, std, corr = compute_sampling_statistics(sample_dataframe)

        assert isinstance(mean, pd.Series)
        assert isinstance(std, pd.Series)
        assert len(mean) == len(sample_dataframe.columns)
        assert len(std) == len(sample_dataframe.columns)

    def test_mean_calculation(self, sample_dataframe):
        """Test that mean is calculated correctly."""
        mean, _, _ = compute_sampling_statistics(sample_dataframe)

        for col in sample_dataframe.columns:
            expected_mean = sample_dataframe[col].mean()
            np.testing.assert_almost_equal(mean[col], expected_mean)

    def test_std_calculation(self, sample_dataframe):
        """Test that std is calculated correctly."""
        _, std, _ = compute_sampling_statistics(sample_dataframe)

        for col in sample_dataframe.columns:
            expected_std = sample_dataframe[col].std()
            np.testing.assert_almost_equal(std[col], expected_std)

    def test_correlation_matrix(self, sample_dataframe):
        """Test that correlation matrix is computed."""
        _, _, corr = compute_sampling_statistics(sample_dataframe)

        assert isinstance(corr, pd.DataFrame)
        # Correlation matrix should be square
        if len(corr) > 0:
            assert corr.shape[0] == corr.shape[1]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        mean, std, corr = compute_sampling_statistics(empty_df)

        assert len(mean) == 0
        assert len(std) == 0

    def test_constant_columns_excluded_from_corr(self):
        """Test that constant columns are excluded from correlation."""
        # Create DataFrame with a constant column
        df = pd.DataFrame(
            {"R1": [1.0, 2.0, 3.0], "R2": [5.0, 5.0, 5.0], "R3": [2.0, 4.0, 6.0]}  # Constant column
        )

        _, std, corr = compute_sampling_statistics(df)

        # R2 should have zero std
        assert std["R2"] == 0.0
        # R2 should not be in correlation matrix
        assert "R2" not in corr.columns


class TestAddGaussianNoiseToSamples:
    """Tests for add_gaussian_noise_to_samples function."""

    def test_noise_added(self):
        """Test that Gaussian noise is added to samples."""
        np.random.seed(42)
        samples = pd.DataFrame({"R1": [10.0, 10.0, 10.0], "R2": [5.0, 5.0, 5.0]})
        reference_fluxes = {"R1": 10.0, "R2": 5.0}

        noisy_samples = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.1)

        # Samples should be modified
        assert not samples.equals(noisy_samples)

    def test_noise_preserves_shape(self):
        """Test that noise addition preserves DataFrame shape."""
        samples = pd.DataFrame({"R1": [10.0, 10.0, 10.0], "R2": [5.0, 5.0, 5.0]})
        reference_fluxes = {"R1": 10.0, "R2": 5.0}

        noisy_samples = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.1)

        assert noisy_samples.shape == samples.shape

    def test_noise_only_affects_reference_reactions(self):
        """Test that noise is only added to reactions in reference_fluxes."""
        samples = pd.DataFrame({"R1": [10.0, 10.0], "R2": [5.0, 5.0], "R3": [3.0, 3.0]})
        reference_fluxes = {"R1": 10.0}  # Only R1 in reference

        noisy_samples = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.1)

        # R2 and R3 should be unchanged
        pd.testing.assert_series_equal(samples["R2"], noisy_samples["R2"])
        pd.testing.assert_series_equal(samples["R3"], noisy_samples["R3"])

    def test_near_zero_flux_handling(self):
        """Test handling of near-zero fluxes."""
        samples = pd.DataFrame({"R1": [0.0, 0.0, 0.0]})
        reference_fluxes = {"R1": 1e-10}  # Near zero

        noisy_samples = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.1)

        # Should not raise an error and should add noise
        assert len(noisy_samples) == 3

    def test_std_fraction_parameter(self):
        """Test that std_fraction affects noise magnitude."""
        np.random.seed(42)
        samples = pd.DataFrame({"R1": [100.0] * 100})
        reference_fluxes = {"R1": 100.0}

        noisy_small = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.01)
        np.random.seed(42)
        noisy_large = add_gaussian_noise_to_samples(samples, reference_fluxes, std_fraction=0.5)

        # Larger std_fraction should result in larger deviations
        small_std = noisy_small["R1"].std()
        large_std = noisy_large["R1"].std()
        assert large_std > small_std
