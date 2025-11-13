"""Unit tests for visualization module."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stock_drawdown.visualization import DrawdownVisualizer  # noqa: E402


class TestDrawdownVisualizer:
    """Test cases for DrawdownVisualizer class."""

    @pytest.fixture
    def sample_drawdowns(self):
        """Create sample drawdown time series."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Asset1': np.random.uniform(-0.2, 0, 100),
            'Asset2': np.random.uniform(-0.15, 0, 100),
            'Asset3': np.random.uniform(-0.25, 0, 100)
        }, index=dates)

    @pytest.fixture
    def sample_magnitudes(self):
        """Create sample drawdown magnitudes."""
        np.random.seed(42)
        return {
            'Asset1': np.random.pareto(2.5, 50) * 0.05 + 0.01,
            'Asset2': np.random.pareto(3.0, 50) * 0.05 + 0.01,
            'Asset3': np.random.pareto(2.0, 50) * 0.05 + 0.01
        }

    @pytest.fixture
    def sample_fitted_params(self):
        """Create sample fitted Pareto parameters."""
        return {
            'Asset1': (2.5, 0, 0.01),
            'Asset2': (3.0, 0, 0.01),
            'Asset3': (2.0, 0, 0.01)
        }

    def test_initialization(self):
        """Test successful initialization."""
        visualizer = DrawdownVisualizer()
        assert visualizer is not None

    def test_initialization_with_custom_style(self):
        """Test initialization with custom style."""
        # Should not raise error even with invalid style (falls back to default)
        visualizer = DrawdownVisualizer(style='invalid_style')
        assert visualizer is not None

    def test_plot_drawdown_series(self, sample_drawdowns):
        """Test plotting drawdown time series."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_drawdown_series(sample_drawdowns)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0

        plt.close(fig)

    def test_plot_drawdown_series_subset(self, sample_drawdowns):
        """Test plotting subset of assets."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_drawdown_series(
            sample_drawdowns,
            asset_names=['Asset1', 'Asset2']
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_drawdown_series_save(self, sample_drawdowns):
        """Test saving drawdown series plot."""
        visualizer = DrawdownVisualizer()

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_plot.png'

            fig = visualizer.plot_drawdown_series(
                sample_drawdowns,
                save_path=str(save_path)
            )

            assert save_path.exists()
            plt.close(fig)

    def test_plot_drawdown_histogram(self, sample_magnitudes):
        """Test plotting drawdown histograms."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_drawdown_histogram(sample_magnitudes)

        assert isinstance(fig, plt.Figure)
        # Should have one subplot per asset
        assert len(fig.axes) >= len(sample_magnitudes)

        plt.close(fig)

    def test_plot_drawdown_histogram_custom_bins(self, sample_magnitudes):
        """Test histogram with custom bins."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_drawdown_histogram(sample_magnitudes, bins=50)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_drawdown_histogram_save(self, sample_magnitudes):
        """Test saving histogram plot."""
        visualizer = DrawdownVisualizer()

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_histogram.png'

            fig = visualizer.plot_drawdown_histogram(
                sample_magnitudes,
                save_path=str(save_path)
            )

            assert save_path.exists()
            plt.close(fig)

    def test_plot_pareto_fit(self, sample_magnitudes, sample_fitted_params):
        """Test plotting Pareto distribution fits."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_pareto_fit(
            sample_magnitudes,
            sample_fitted_params
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pareto_fit_with_nan_params(self, sample_magnitudes):
        """Test plotting with NaN fitted parameters."""
        visualizer = DrawdownVisualizer()

        params_with_nan = {
            'Asset1': (2.5, 0, 0.01),
            'Asset2': (np.nan, np.nan, np.nan),
            'Asset3': (2.0, 0, 0.01)
        }

        fig = visualizer.plot_pareto_fit(
            sample_magnitudes,
            params_with_nan
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pareto_fit_log(self, sample_magnitudes, sample_fitted_params):
        """Test plotting log-log Pareto fits."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_pareto_fit_log(
            sample_magnitudes,
            sample_fitted_params
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_qq(self, sample_magnitudes, sample_fitted_params):
        """Test plotting QQ plots."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.plot_qq(
            sample_magnitudes,
            sample_fitted_params
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_qq_with_nan_params(self, sample_magnitudes):
        """Test QQ plot with NaN parameters."""
        visualizer = DrawdownVisualizer()

        params_with_nan = {
            'Asset1': (2.5, 0, 0.01),
            'Asset2': (np.nan, np.nan, np.nan)
        }

        fig = visualizer.plot_qq(
            {'Asset1': sample_magnitudes['Asset1'],
             'Asset2': sample_magnitudes['Asset2']},
            params_with_nan
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_qq_missing_params(self, sample_magnitudes):
        """Test QQ plot with missing fitted params."""
        visualizer = DrawdownVisualizer()

        # Only provide params for Asset1
        partial_params = {
            'Asset1': (2.5, 0, 0.01)
        }

        fig = visualizer.plot_qq(
            sample_magnitudes,
            partial_params
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_summary_report(
        self,
        sample_magnitudes,
        sample_fitted_params
    ):
        """Test creating comprehensive summary report."""
        visualizer = DrawdownVisualizer()

        fig = visualizer.create_summary_report(
            sample_magnitudes,
            sample_fitted_params
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_summary_report_with_gof(
        self,
        sample_magnitudes,
        sample_fitted_params
    ):
        """Test summary report with GOF statistics."""
        visualizer = DrawdownVisualizer()

        gof_stats = {
            'Asset1': {
                'ks_statistic': 0.05,
                'ks_pvalue': 0.85,
                'ad_statistic': 0.3,
                'log_likelihood': -50.0
            },
            'Asset2': {
                'ks_statistic': 0.06,
                'ks_pvalue': 0.75,
                'ad_statistic': 0.4,
                'log_likelihood': -55.0
            },
            'Asset3': {
                'ks_statistic': 0.07,
                'ks_pvalue': 0.70,
                'ad_statistic': 0.5,
                'log_likelihood': -60.0
            }
        }

        fig = visualizer.create_summary_report(
            sample_magnitudes,
            sample_fitted_params,
            goodness_of_fit=gof_stats
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_summary_report_save(
        self,
        sample_magnitudes,
        sample_fitted_params
    ):
        """Test saving summary report."""
        visualizer = DrawdownVisualizer()

        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'summary.png'

            fig = visualizer.create_summary_report(
                sample_magnitudes,
                sample_fitted_params,
                save_path=str(save_path)
            )

            assert save_path.exists()
            plt.close(fig)

    def test_single_asset_visualization(self):
        """Test visualizations with single asset."""
        visualizer = DrawdownVisualizer()

        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        drawdowns = pd.DataFrame({
            'Asset1': np.random.uniform(-0.2, 0, 50)
        }, index=dates)

        np.random.seed(42)
        magnitudes = {
            'Asset1': np.random.pareto(2.5, 30) * 0.05 + 0.01
        }

        params = {'Asset1': (2.5, 0, 0.01)}

        # Test all plot types with single asset
        fig1 = visualizer.plot_drawdown_series(drawdowns)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = visualizer.plot_drawdown_histogram(magnitudes)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        fig3 = visualizer.plot_pareto_fit(magnitudes, params)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

        fig4 = visualizer.plot_pareto_fit_log(magnitudes, params)
        assert isinstance(fig4, plt.Figure)
        plt.close(fig4)

        fig5 = visualizer.plot_qq(magnitudes, params)
        assert isinstance(fig5, plt.Figure)
        plt.close(fig5)

    def test_many_assets_layout(self):
        """Test that layout works with many assets."""
        visualizer = DrawdownVisualizer()

        np.random.seed(42)
        magnitudes = {
            f'Asset{i}': np.random.pareto(2.0 + i * 0.1, 30) * 0.05 + 0.01
            for i in range(1, 8)  # 7 assets
        }

        params = {
            f'Asset{i}': (2.0 + i * 0.1, 0, 0.01)
            for i in range(1, 8)
        }

        fig = visualizer.plot_pareto_fit(magnitudes, params)
        assert isinstance(fig, plt.Figure)

        # Should have at least 7 axes (one per asset)
        assert len(fig.axes) >= 7

        plt.close(fig)

    def test_empty_asset_list(self, sample_drawdowns):
        """Test with empty asset list."""
        visualizer = DrawdownVisualizer()

        # Should plot nothing or all assets
        fig = visualizer.plot_drawdown_series(sample_drawdowns, asset_names=[])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
