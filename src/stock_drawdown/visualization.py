"""Module for visualizing drawdown data and fitted distributions."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class DrawdownVisualizer:
    """Create visualizations for drawdown analysis.

    This class provides methods to visualize:
    - Drawdown time series
    - Histograms of drawdown magnitudes
    - Fitted Pareto distributions
    - QQ plots for goodness of fit
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize the DrawdownVisualizer.

        Args:
            style: Matplotlib style to use for plots.
        """
        try:
            plt.style.use(style)
        except Exception:
            # Fallback to default if style not available
            plt.style.use('default')

        # Set seaborn color palette
        sns.set_palette("husl")

        # Configure matplotlib for better-looking plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10

    def plot_drawdown_series(
        self,
        drawdowns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot drawdown time series for selected assets.

        Args:
            drawdowns: DataFrame with drawdown values (columns are assets).
            asset_names: List of asset names to plot. If None, plots all.
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        if asset_names is None:
            asset_names = drawdowns.columns.tolist()

        fig, ax = plt.subplots(figsize=(14, 6))

        for asset in asset_names:
            if asset in drawdowns.columns:
                # Convert to percentage for display
                drawdown_pct = drawdowns[asset] * 100
                ax.plot(
                    drawdown_pct.index,
                    drawdown_pct,
                    label=asset,
                    linewidth=1.5,
                    alpha=0.8
                )

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Time Series by Index')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_drawdown_histogram(
        self,
        drawdown_magnitudes: Dict[str, np.ndarray],
        bins: int = 30,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot histograms of drawdown magnitudes for all assets.

        Args:
            drawdown_magnitudes: Dictionary mapping asset names to magnitude arrays.
            bins: Number of histogram bins.
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        n_assets = len(drawdown_magnitudes)
        n_cols = min(3, n_assets)
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_assets == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, (asset, magnitudes) in enumerate(drawdown_magnitudes.items()):
            ax = axes[idx]

            # Convert to percentage for display
            magnitudes_pct = magnitudes * 100

            ax.hist(
                magnitudes_pct,
                bins=bins,
                density=True,
                alpha=0.7,
                color='steelblue',
                edgecolor='black'
            )

            ax.set_xlabel('Drawdown Magnitude (%)')
            ax.set_ylabel('Density')
            ax.set_title(f'{asset} - Drawdown Distribution')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = (
                f'Count: {len(magnitudes)}\n'
                f'Mean: {np.mean(magnitudes_pct):.2f}%\n'
                f'Median: {np.median(magnitudes_pct):.2f}%\n'
                f'Max: {np.max(magnitudes_pct):.2f}%'
            )
            ax.text(
                0.95, 0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8
            )

        # Hide unused subplots
        for idx in range(n_assets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_pareto_fit(
        self,
        drawdown_magnitudes: Dict[str, np.ndarray],
        fitted_params: Dict[str, Tuple[float, float, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot drawdown histograms with fitted Pareto distributions.

        Args:
            drawdown_magnitudes: Dictionary of drawdown magnitude arrays.
            fitted_params: Dictionary of fitted Pareto parameters (shape, loc, scale).
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        n_assets = len(drawdown_magnitudes)
        n_cols = min(3, n_assets)
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_assets == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, asset in enumerate(drawdown_magnitudes.keys()):
            ax = axes[idx]

            magnitudes = drawdown_magnitudes[asset]
            magnitudes_pct = magnitudes * 100

            # Plot histogram
            ax.hist(
                magnitudes_pct,
                bins=30,
                density=True,
                alpha=0.6,
                color='steelblue',
                edgecolor='black',
                label='Observed'
            )

            # Plot fitted Pareto distribution
            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]

                if not np.isnan(shape):
                    # Generate x values for the PDF
                    x_min = scale
                    x_max = max(magnitudes) * 1.2
                    x = np.linspace(x_min, x_max, 1000)

                    # Calculate PDF (convert scale to percentage)
                    pdf = stats.pareto.pdf(x / 100, shape, loc, scale) / 100

                    ax.plot(
                        x * 100,
                        pdf,
                        'r-',
                        linewidth=2,
                        label=f'Pareto Fit (α={shape:.2f})'
                    )

            ax.set_xlabel('Drawdown Magnitude (%)')
            ax.set_ylabel('Density')
            ax.set_title(f'{asset} - Pareto Distribution Fit')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_assets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_pareto_fit_log(
        self,
        drawdown_magnitudes: Dict[str, np.ndarray],
        fitted_params: Dict[str, Tuple[float, float, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot log-log plot of drawdown distributions with Pareto fit.

        The Pareto distribution appears as a straight line on a log-log plot,
        which makes it easier to assess the quality of the fit.

        Args:
            drawdown_magnitudes: Dictionary of drawdown magnitude arrays.
            fitted_params: Dictionary of fitted Pareto parameters.
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        n_assets = len(drawdown_magnitudes)
        n_cols = min(3, n_assets)
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_assets == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, asset in enumerate(drawdown_magnitudes.keys()):
            ax = axes[idx]

            magnitudes = drawdown_magnitudes[asset]

            # Calculate empirical survival function (complementary CDF)
            sorted_mags = np.sort(magnitudes)
            n = len(sorted_mags)
            survival = 1 - np.arange(1, n + 1) / (n + 1)

            # Plot empirical data
            ax.loglog(
                sorted_mags * 100,
                survival,
                'o',
                markersize=4,
                alpha=0.6,
                label='Observed'
            )

            # Plot fitted Pareto survival function
            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]

                if not np.isnan(shape):
                    x = np.logspace(
                        np.log10(scale),
                        np.log10(max(magnitudes) * 1.2),
                        100
                    )

                    # Survival function
                    sf = stats.pareto.sf(x, shape, loc, scale)

                    ax.loglog(
                        x * 100,
                        sf,
                        'r-',
                        linewidth=2,
                        label=f'Pareto Fit (α={shape:.2f})'
                    )

            ax.set_xlabel('Drawdown Magnitude (%, log scale)')
            ax.set_ylabel('P(X > x) (log scale)')
            ax.set_title(f'{asset} - Log-Log Plot')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, which='both')

        # Hide unused subplots
        for idx in range(n_assets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def plot_qq(
        self,
        drawdown_magnitudes: Dict[str, np.ndarray],
        fitted_params: Dict[str, Tuple[float, float, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create QQ plots to assess goodness of fit.

        Args:
            drawdown_magnitudes: Dictionary of drawdown magnitude arrays.
            fitted_params: Dictionary of fitted Pareto parameters.
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        n_assets = len(drawdown_magnitudes)
        n_cols = min(3, n_assets)
        n_rows = (n_assets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n_assets == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, asset in enumerate(drawdown_magnitudes.keys()):
            ax = axes[idx]

            magnitudes = drawdown_magnitudes[asset]

            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]

                if not np.isnan(shape):
                    # Generate theoretical quantiles
                    sorted_mags = np.sort(magnitudes)
                    n = len(sorted_mags)
                    theoretical_quantiles = stats.pareto.ppf(
                        np.linspace(0.01, 0.99, n),
                        shape,
                        loc,
                        scale
                    )

                    # Plot QQ plot
                    ax.scatter(
                        theoretical_quantiles * 100,
                        sorted_mags * 100,
                        alpha=0.6,
                        s=20
                    )

                    # Plot reference line
                    min_val = min(
                        theoretical_quantiles.min(),
                        sorted_mags.min()
                    ) * 100
                    max_val = max(
                        theoretical_quantiles.max(),
                        sorted_mags.max()
                    ) * 100

                    ax.plot(
                        [min_val, max_val],
                        [min_val, max_val],
                        'r--',
                        linewidth=2,
                        label='Perfect Fit'
                    )

                    ax.set_xlabel('Theoretical Quantiles (%)')
                    ax.set_ylabel('Observed Quantiles (%)')
                    ax.set_title(f'{asset} - QQ Plot')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(
                        0.5, 0.5,
                        'Fit Failed',
                        transform=ax.transAxes,
                        ha='center',
                        va='center'
                    )
                    ax.set_title(f'{asset} - QQ Plot')
            else:
                ax.text(
                    0.5, 0.5,
                    'No Fit Available',
                    transform=ax.transAxes,
                    ha='center',
                    va='center'
                )
                ax.set_title(f'{asset} - QQ Plot')

        # Hide unused subplots
        for idx in range(n_assets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def create_summary_report(
        self,
        drawdown_magnitudes: Dict[str, np.ndarray],
        fitted_params: Dict[str, Tuple[float, float, float]],
        goodness_of_fit: Optional[Dict[str, Dict[str, float]]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create a comprehensive summary report with multiple plots.

        Args:
            drawdown_magnitudes: Dictionary of drawdown magnitude arrays.
            fitted_params: Dictionary of fitted Pareto parameters.
            goodness_of_fit: Optional dictionary of GOF statistics.
            save_path: Path to save the figure. If None, doesn't save.

        Returns:
            Matplotlib Figure object.
        """
        n_assets = len(drawdown_magnitudes)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, n_assets, hspace=0.3, wspace=0.3)

        for idx, asset in enumerate(drawdown_magnitudes.keys()):
            magnitudes = drawdown_magnitudes[asset]
            magnitudes_pct = magnitudes * 100

            # Row 1: Histogram with fit
            ax1 = fig.add_subplot(gs[0, idx])
            ax1.hist(
                magnitudes_pct,
                bins=30,
                density=True,
                alpha=0.6,
                color='steelblue',
                edgecolor='black'
            )

            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]
                if not np.isnan(shape):
                    x = np.linspace(scale, max(magnitudes) * 1.2, 1000)
                    pdf = stats.pareto.pdf(x / 100, shape, loc, scale) / 100
                    ax1.plot(
                        x * 100,
                        pdf,
                        'r-',
                        linewidth=2,
                        label=f'α={shape:.2f}'
                    )
                    ax1.legend()

            ax1.set_xlabel('Drawdown (%)')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{asset}')
            ax1.grid(True, alpha=0.3)

            # Row 2: Log-log plot
            ax2 = fig.add_subplot(gs[1, idx])
            sorted_mags = np.sort(magnitudes)
            n = len(sorted_mags)
            survival = 1 - np.arange(1, n + 1) / (n + 1)

            ax2.loglog(
                sorted_mags * 100,
                survival,
                'o',
                markersize=3,
                alpha=0.6
            )

            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]
                if not np.isnan(shape):
                    x = np.logspace(
                        np.log10(scale),
                        np.log10(max(magnitudes) * 1.2),
                        100
                    )
                    sf = stats.pareto.sf(x, shape, loc, scale)
                    ax2.loglog(x * 100, sf, 'r-', linewidth=2)

            ax2.set_xlabel('Drawdown (%, log)')
            ax2.set_ylabel('P(X > x) (log)')
            ax2.grid(True, alpha=0.3, which='both')

            # Row 3: QQ plot
            ax3 = fig.add_subplot(gs[2, idx])

            if asset in fitted_params:
                shape, loc, scale = fitted_params[asset]
                if not np.isnan(shape):
                    theoretical = stats.pareto.ppf(
                        np.linspace(0.01, 0.99, n),
                        shape,
                        loc,
                        scale
                    )
                    ax3.scatter(
                        theoretical * 100,
                        sorted_mags * 100,
                        alpha=0.6,
                        s=10
                    )

                    min_val = min(theoretical.min(), sorted_mags.min()) * 100
                    max_val = max(theoretical.max(), sorted_mags.max()) * 100
                    ax3.plot(
                        [min_val, max_val],
                        [min_val, max_val],
                        'r--',
                        linewidth=2
                    )

            ax3.set_xlabel('Theoretical (%)')
            ax3.set_ylabel('Observed (%)')
            ax3.grid(True, alpha=0.3)

            # Add GOF statistics if available
            if goodness_of_fit and asset in goodness_of_fit:
                gof = goodness_of_fit[asset]
                stats_text = (
                    f"KS stat: {gof['ks_statistic']:.3f}\n"
                    f"p-value: {gof['ks_pvalue']:.3f}"
                )
                ax3.text(
                    0.05, 0.95,
                    stats_text,
                    transform=ax3.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=7
                )

        plt.suptitle(
            'Drawdown Analysis - Pareto Distribution Fits',
            fontsize=14,
            y=0.995
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig
