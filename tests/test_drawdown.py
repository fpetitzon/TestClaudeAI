"""Unit tests for drawdown calculation module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stock_drawdown.drawdown import DrawdownCalculator  # noqa: E402


class TestDrawdownCalculator:
    """Test cases for DrawdownCalculator class."""

    @pytest.fixture
    def simple_price_data(self):
        """Create simple price data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Asset1': [100, 110, 105, 95, 90, 100, 105, 95, 90, 100],
            'Asset2': [200, 210, 220, 215, 210, 205, 200, 210, 220, 230]
        }, index=dates)
        return data

    @pytest.fixture
    def monotonic_price_data(self):
        """Create monotonically increasing price data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Asset1': np.linspace(100, 200, 10)
        }, index=dates)
        return data

    def test_initialization_success(self, simple_price_data):
        """Test successful initialization."""
        calculator = DrawdownCalculator(simple_price_data)
        assert calculator.price_data is not None
        assert len(calculator.price_data) == 10
        assert list(calculator.price_data.columns) == ['Asset1', 'Asset2']

    def test_initialization_with_empty_data(self):
        """Test that empty data raises ValueError."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            DrawdownCalculator(empty_data)

    def test_initialization_with_non_positive_values(self, simple_price_data):
        """Test that non-positive values raise ValueError."""
        simple_price_data.iloc[0, 0] = 0
        with pytest.raises(ValueError, match="non-positive values"):
            DrawdownCalculator(simple_price_data)

        simple_price_data.iloc[0, 0] = -10
        with pytest.raises(ValueError, match="non-positive values"):
            DrawdownCalculator(simple_price_data)

    def test_initialization_with_nan_values(self, simple_price_data):
        """Test that NaN values raise ValueError."""
        simple_price_data.iloc[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            DrawdownCalculator(simple_price_data)

    def test_calculate_drawdowns_basic(self, simple_price_data):
        """Test basic drawdown calculation."""
        calculator = DrawdownCalculator(simple_price_data)
        drawdowns = calculator.calculate_drawdowns()

        assert isinstance(drawdowns, pd.DataFrame)
        assert drawdowns.shape == simple_price_data.shape
        assert (drawdowns <= 0).all().all()  # All drawdowns should be non-positive

    def test_calculate_drawdowns_monotonic(self, monotonic_price_data):
        """Test drawdowns for monotonically increasing prices."""
        calculator = DrawdownCalculator(monotonic_price_data)
        drawdowns = calculator.calculate_drawdowns()

        # Should have all zeros (no drawdown)
        assert (drawdowns == 0).all().all()

    def test_calculate_drawdowns_values(self):
        """Test drawdown calculation with known values."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Asset': [100, 90, 80, 90, 100]
        }, index=dates)

        calculator = DrawdownCalculator(data)
        drawdowns = calculator.calculate_drawdowns()

        # Expected drawdowns
        expected = pd.Series([0.0, -0.1, -0.2, -0.1, 0.0], index=dates)

        np.testing.assert_array_almost_equal(
            drawdowns['Asset'].values,
            expected.values,
            decimal=10
        )

    def test_get_drawdown_periods_simple(self):
        """Test identification of drawdown periods."""
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        data = pd.DataFrame({
            'Asset': [100, 90, 85, 95, 100, 95, 100]
        }, index=dates)

        calculator = DrawdownCalculator(data)
        periods = calculator.get_drawdown_periods(min_drawdown=0.05)

        assert 'Asset' in periods
        assert len(periods['Asset']) >= 1

        # Check first period
        first_period = periods['Asset'][0]
        assert 'start' in first_period
        assert 'end' in first_period
        assert 'peak' in first_period
        assert 'trough' in first_period
        assert 'max_drawdown' in first_period
        assert first_period['max_drawdown'] >= 0.05

    def test_get_drawdown_periods_minimum_filter(self):
        """Test that minimum drawdown filter works."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Asset': [100, 98, 97, 99, 100]  # Small 3% drawdown
        }, index=dates)

        calculator = DrawdownCalculator(data)

        # With high minimum, should get no periods
        periods_high = calculator.get_drawdown_periods(min_drawdown=0.05)
        assert len(periods_high['Asset']) == 0

        # With low minimum, should get the period
        periods_low = calculator.get_drawdown_periods(min_drawdown=0.01)
        assert len(periods_low['Asset']) > 0

    def test_get_drawdown_periods_no_recovery(self):
        """Test drawdown period that doesn't recover."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Asset': [100, 95, 90, 85, 80]  # Continuous decline
        }, index=dates)

        calculator = DrawdownCalculator(data)
        periods = calculator.get_drawdown_periods(min_drawdown=0.05)

        assert len(periods['Asset']) == 1

        period = periods['Asset'][0]
        assert period['end'] == dates[-1]  # Ends at last date
        assert period['max_drawdown'] == 0.20  # 20% drawdown

    def test_get_all_drawdown_magnitudes(self, simple_price_data):
        """Test extraction of all drawdown magnitudes."""
        calculator = DrawdownCalculator(simple_price_data)
        magnitudes = calculator.get_all_drawdown_magnitudes(min_drawdown=0.01)

        assert isinstance(magnitudes, dict)
        assert 'Asset1' in magnitudes
        assert 'Asset2' in magnitudes

        # Magnitudes should be positive
        for asset, mags in magnitudes.items():
            assert (mags >= 0).all()

    def test_get_summary_statistics(self, simple_price_data):
        """Test summary statistics calculation."""
        calculator = DrawdownCalculator(simple_price_data)
        stats = calculator.get_summary_statistics(min_drawdown=0.01)

        assert isinstance(stats, pd.DataFrame)
        assert 'Asset1' in stats.index
        assert 'Asset2' in stats.index

        # Check expected columns
        expected_cols = ['count', 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for col in expected_cols:
            assert col in stats.columns

        # Count should be positive integers
        assert (stats['count'] >= 0).all()

    def test_get_summary_statistics_no_drawdowns(self, monotonic_price_data):
        """Test summary statistics with no drawdowns."""
        calculator = DrawdownCalculator(monotonic_price_data)
        stats = calculator.get_summary_statistics(min_drawdown=0.01)

        # Should have zero count
        assert stats.loc['Asset1', 'count'] == 0
        assert pd.isna(stats.loc['Asset1', 'mean'])

    def test_get_worst_drawdowns(self):
        """Test getting worst drawdowns."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        # Create data with multiple drawdowns of different magnitudes
        data = pd.DataFrame({
            'Asset': [
                100, 90, 85, 90, 100,  # 15% drawdown
                95, 90, 100,            # 10% drawdown
                95, 85, 90, 100,        # 15% drawdown
                90, 80, 75, 90, 100,    # 25% drawdown
                95, 90, 100             # 10% drawdown
            ]
        }, index=dates)

        calculator = DrawdownCalculator(data)
        worst = calculator.get_worst_drawdowns(n=3, min_drawdown=0.05)

        assert 'Asset' in worst
        assert len(worst['Asset']) <= 3

        # Check they are sorted by magnitude
        for i in range(len(worst['Asset']) - 1):
            assert (worst['Asset'][i]['max_drawdown'] >=
                    worst['Asset'][i + 1]['max_drawdown'])

        # Worst should be 25%
        assert worst['Asset'][0]['max_drawdown'] == 0.25

    def test_multiple_assets(self):
        """Test with multiple assets."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Asset1': [100, 90, 95, 100, 105, 100, 95, 100, 105, 110],
            'Asset2': [200, 190, 180, 185, 190, 200, 210, 205, 210, 220],
            'Asset3': [300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
        }, index=dates)

        calculator = DrawdownCalculator(data)
        drawdowns = calculator.calculate_drawdowns()

        assert len(drawdowns.columns) == 3
        assert 'Asset1' in drawdowns.columns
        assert 'Asset2' in drawdowns.columns
        assert 'Asset3' in drawdowns.columns

        # Asset3 should have no drawdowns (monotonic)
        assert (drawdowns['Asset3'] == 0).all()

    def test_drawdown_calculation_idempotent(self, simple_price_data):
        """Test that multiple calls to calculate_drawdowns give same result."""
        calculator = DrawdownCalculator(simple_price_data)

        drawdowns1 = calculator.calculate_drawdowns()
        drawdowns2 = calculator.calculate_drawdowns()

        pd.testing.assert_frame_equal(drawdowns1, drawdowns2)

    def test_period_duration(self):
        """Test that period duration is calculated correctly."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Asset': [100, 95, 90, 85, 90, 95, 100, 95, 90, 100]
        }, index=dates)

        calculator = DrawdownCalculator(data)
        periods = calculator.get_drawdown_periods(min_drawdown=0.05)

        for period in periods['Asset']:
            assert period['duration_days'] > 0
            assert period['duration_days'] <= len(dates)
