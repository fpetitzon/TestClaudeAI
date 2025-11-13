"""Unit tests for Pareto distribution fitting module."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stock_drawdown.distribution import ParetoDistributionFitter  # noqa: E402


class TestParetoDistributionFitter:
    """Test cases for ParetoDistributionFitter class."""

    @pytest.fixture
    def simple_magnitudes(self):
        """Create simple drawdown magnitudes for testing."""
        np.random.seed(42)
        return {
            'Asset1': np.random.pareto(2.5, 50) * 0.05 + 0.01,
            'Asset2': np.random.pareto(3.0, 50) * 0.05 + 0.01
        }

    @pytest.fixture
    def single_asset_magnitudes(self):
        """Create magnitudes for a single asset."""
        np.random.seed(42)
        return {
            'Asset1': np.random.pareto(2.0, 100) * 0.05 + 0.01
        }

    def test_initialization_success(self, simple_magnitudes):
        """Test successful initialization."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        assert fitter.drawdown_magnitudes is not None
        assert 'Asset1' in fitter.drawdown_magnitudes
        assert 'Asset2' in fitter.drawdown_magnitudes

    def test_initialization_with_empty_dict(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ParetoDistributionFitter({})

    def test_initialization_with_empty_array(self):
        """Test that empty array for asset raises ValueError."""
        with pytest.raises(ValueError, match="No drawdowns"):
            ParetoDistributionFitter({'Asset1': np.array([])})

    def test_initialization_with_non_positive_values(self):
        """Test that non-positive values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParetoDistributionFitter({'Asset1': np.array([0.1, 0.2, 0.0])})

        with pytest.raises(ValueError, match="must be positive"):
            ParetoDistributionFitter({'Asset1': np.array([0.1, -0.2, 0.3])})

    def test_fit_pareto_mle(self, simple_magnitudes):
        """Test Pareto fitting with MLE method."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        shape, loc, scale = fitter.fit_pareto('Asset1', method='mle')

        assert shape > 0
        assert loc == 0  # Should be fixed at 0
        assert scale > 0
        assert scale <= np.min(simple_magnitudes['Asset1'])

    def test_fit_pareto_moments(self, simple_magnitudes):
        """Test Pareto fitting with moments method."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        shape, loc, scale = fitter.fit_pareto('Asset1', method='moments')

        assert shape > 0
        assert loc == 0
        assert scale > 0

    def test_fit_pareto_invalid_method(self, simple_magnitudes):
        """Test that invalid method raises ValueError."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        with pytest.raises(ValueError, match="Invalid method"):
            fitter.fit_pareto('Asset1', method='invalid')

    def test_fit_pareto_invalid_asset(self, simple_magnitudes):
        """Test that invalid asset name raises ValueError."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        with pytest.raises(ValueError, match="Asset not found"):
            fitter.fit_pareto('InvalidAsset')

    def test_fit_all_assets(self, simple_magnitudes):
        """Test fitting all assets."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        params = fitter.fit_all_assets(method='mle')

        assert isinstance(params, dict)
        assert 'Asset1' in params
        assert 'Asset2' in params

        for asset, (shape, loc, scale) in params.items():
            assert shape > 0
            assert loc == 0
            assert scale > 0

    def test_fit_stores_params(self, simple_magnitudes):
        """Test that fitting stores parameters."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        # Should be able to get fitted params
        params = fitter.get_fitted_params('Asset1')
        assert params is not None
        assert len(params) == 3

    def test_get_fitted_params_before_fitting(self, simple_magnitudes):
        """Test that getting params before fitting raises error."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        with pytest.raises(RuntimeError, match="not fitted"):
            fitter.get_fitted_params('Asset1')

    def test_get_all_fitted_params(self, simple_magnitudes):
        """Test getting all fitted parameters."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_all_assets()

        all_params = fitter.get_all_fitted_params()

        assert isinstance(all_params, dict)
        assert len(all_params) == 2
        assert 'Asset1' in all_params
        assert 'Asset2' in all_params

    def test_calculate_goodness_of_fit(self, simple_magnitudes):
        """Test goodness-of-fit calculation."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        gof = fitter.calculate_goodness_of_fit('Asset1')

        assert isinstance(gof, dict)
        assert 'ks_statistic' in gof
        assert 'ks_pvalue' in gof
        assert 'ad_statistic' in gof
        assert 'log_likelihood' in gof

        # Check that values are reasonable
        assert 0 <= gof['ks_statistic'] <= 1
        assert 0 <= gof['ks_pvalue'] <= 1
        assert gof['ad_statistic'] >= 0

    def test_calculate_goodness_of_fit_before_fitting(self, simple_magnitudes):
        """Test GOF calculation before fitting raises error."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        with pytest.raises(RuntimeError, match="not fitted"):
            fitter.calculate_goodness_of_fit('Asset1')

    def test_calculate_all_goodness_of_fit(self, simple_magnitudes):
        """Test GOF calculation for all assets."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_all_assets()

        gof_all = fitter.calculate_all_goodness_of_fit()

        assert isinstance(gof_all, dict)
        assert 'Asset1' in gof_all
        assert 'Asset2' in gof_all

        for asset, gof in gof_all.items():
            assert 'ks_statistic' in gof
            assert 'ks_pvalue' in gof

    def test_calculate_all_goodness_of_fit_before_fitting(self, simple_magnitudes):
        """Test GOF for all before fitting raises error."""
        fitter = ParetoDistributionFitter(simple_magnitudes)

        with pytest.raises(RuntimeError, match="No fitted distributions"):
            fitter.calculate_all_goodness_of_fit()

    def test_calculate_tail_probability(self, simple_magnitudes):
        """Test tail probability calculation."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        # Calculate probability of exceeding 10% drawdown
        prob = fitter.calculate_tail_probability('Asset1', 0.10)

        assert 0 <= prob <= 1

        # Higher threshold should have lower probability
        prob_high = fitter.calculate_tail_probability('Asset1', 0.20)
        assert prob_high < prob

    def test_calculate_tail_probability_invalid_threshold(self, simple_magnitudes):
        """Test that invalid threshold raises error."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        with pytest.raises(ValueError, match="must be positive"):
            fitter.calculate_tail_probability('Asset1', 0)

        with pytest.raises(ValueError, match="must be positive"):
            fitter.calculate_tail_probability('Asset1', -0.1)

    def test_calculate_value_at_risk(self, simple_magnitudes):
        """Test VaR calculation."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        # Calculate 95% VaR
        var_95 = fitter.calculate_value_at_risk('Asset1', 0.95)

        assert var_95 > 0

        # 99% VaR should be higher than 95% VaR
        var_99 = fitter.calculate_value_at_risk('Asset1', 0.99)
        assert var_99 > var_95

    def test_calculate_value_at_risk_invalid_confidence(self, simple_magnitudes):
        """Test that invalid confidence raises error."""
        fitter = ParetoDistributionFitter(simple_magnitudes)
        fitter.fit_pareto('Asset1')

        with pytest.raises(ValueError, match="between 0 and 1"):
            fitter.calculate_value_at_risk('Asset1', 0)

        with pytest.raises(ValueError, match="between 0 and 1"):
            fitter.calculate_value_at_risk('Asset1', 1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            fitter.calculate_value_at_risk('Asset1', 1.5)

    def test_fit_quality_with_known_distribution(self):
        """Test fitting quality with data from known Pareto distribution."""
        # Generate data from known Pareto distribution
        np.random.seed(42)
        true_shape = 2.5
        true_scale = 0.01

        # Generate Pareto samples
        data = stats.pareto.rvs(true_shape, loc=0, scale=true_scale, size=1000)

        fitter = ParetoDistributionFitter({'Asset': data})
        shape, loc, scale = fitter.fit_pareto('Asset', method='mle')

        # Fitted parameters should be close to true parameters
        assert abs(shape - true_shape) < 0.5  # Allow some tolerance
        assert abs(scale - true_scale) < 0.01

    def test_moments_method_fallback(self):
        """Test that moments method falls back to MLE if needed."""
        # Create data where mean <= min (edge case)
        data = np.array([0.10] * 10)  # All same values

        fitter = ParetoDistributionFitter({'Asset': data})

        # Should not raise error, should fall back to MLE
        shape, loc, scale = fitter.fit_pareto('Asset', method='moments')

        assert shape > 0
        assert scale > 0

    def test_single_asset(self, single_asset_magnitudes):
        """Test with single asset."""
        fitter = ParetoDistributionFitter(single_asset_magnitudes)
        params = fitter.fit_all_assets()

        assert len(params) == 1
        assert 'Asset1' in params

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        large_data = {
            'Asset': np.random.pareto(2.0, 10000) * 0.05 + 0.01
        }

        fitter = ParetoDistributionFitter(large_data)
        shape, loc, scale = fitter.fit_pareto('Asset')

        assert shape > 0
        assert scale > 0

        # With large dataset, GOF should be good
        gof = fitter.calculate_goodness_of_fit('Asset')
        # KS p-value should be reasonable (not too small)
        assert gof['ks_pvalue'] > 0.001  # Relaxed threshold
