"""Module for calculating drawdowns in financial time series."""

from typing import Dict, List

import numpy as np
import pandas as pd


class DrawdownCalculator:
    """Calculate drawdowns for financial time series data.

    A drawdown is the peak-to-trough decline during a specific period
    for an investment. This class provides methods to calculate drawdowns,
    identify drawdown periods, and analyze their severity.
    """

    def __init__(self, price_data: pd.DataFrame):
        """Initialize the DrawdownCalculator.

        Args:
            price_data: DataFrame with close prices. Each column represents
                a different index/asset, and the index is the date.

        Raises:
            ValueError: If price_data is empty or contains invalid values.
        """
        if price_data.empty:
            raise ValueError("price_data cannot be empty")

        if (price_data <= 0).any().any():
            raise ValueError("price_data contains non-positive values")

        if price_data.isnull().any().any():
            raise ValueError("price_data contains NaN values")

        self.price_data = price_data.copy()
        self._drawdowns = None
        self._drawdown_periods = None

    def calculate_drawdowns(self) -> pd.DataFrame:
        """Calculate drawdown series for all assets.

        Drawdown at time t is defined as:
            DD(t) = (Price(t) - RunningMax(t)) / RunningMax(t)

        Returns:
            DataFrame with drawdown values (negative percentages).
            Shape matches input price_data.
        """
        # Calculate running maximum
        running_max = self.price_data.expanding(min_periods=1).max()

        # Calculate drawdown as percentage
        drawdowns = (self.price_data - running_max) / running_max

        # Store for future use
        self._drawdowns = drawdowns

        return drawdowns

    def get_drawdown_periods(
        self,
        min_drawdown: float = 0.01
    ) -> Dict[str, List[Dict]]:
        """Identify individual drawdown periods for each asset.

        A drawdown period starts when price falls below previous peak
        and ends when price recovers to or exceeds that peak.

        Args:
            min_drawdown: Minimum drawdown magnitude (as decimal) to include.
                Default 0.01 means 1% drawdown.

        Returns:
            Dictionary mapping asset names to lists of drawdown periods.
            Each period is a dict with keys:
                - start: Start date of drawdown
                - end: End date (recovery) of drawdown
                - peak: Peak price before drawdown
                - trough: Lowest price during drawdown
                - trough_date: Date of trough
                - max_drawdown: Maximum drawdown magnitude (positive decimal)
                - duration_days: Number of trading days

        Raises:
            RuntimeError: If drawdowns have not been calculated yet.
        """
        if self._drawdowns is None:
            self.calculate_drawdowns()

        all_periods = {}

        for column in self.price_data.columns:
            prices = self.price_data[column]
            periods = self._identify_periods_for_asset(
                prices,
                column,
                min_drawdown
            )
            all_periods[column] = periods

        self._drawdown_periods = all_periods
        return all_periods

    def _identify_periods_for_asset(
        self,
        prices: pd.Series,
        asset_name: str,
        min_drawdown: float
    ) -> List[Dict]:
        """Identify drawdown periods for a single asset.

        Args:
            prices: Series of prices for one asset.
            asset_name: Name of the asset.
            min_drawdown: Minimum drawdown magnitude to include.

        Returns:
            List of dictionaries describing each drawdown period.
        """
        periods = []
        running_max = prices.iloc[0]
        peak_date = prices.index[0]
        in_drawdown = False
        current_period = None

        for date, price in prices.items():
            # Update running maximum
            if price >= running_max:
                # Recovered or new high
                if in_drawdown and current_period is not None:
                    # End of drawdown period
                    current_period['end'] = date
                    current_period['duration_days'] = len(
                        prices[
                            current_period['start']:current_period['end']
                        ]
                    )

                    # Only include if significant enough
                    if current_period['max_drawdown'] >= min_drawdown:
                        periods.append(current_period)

                    in_drawdown = False
                    current_period = None

                running_max = price
                peak_date = date
            else:
                # In drawdown
                drawdown_pct = (price - running_max) / running_max
                drawdown_magnitude = abs(drawdown_pct)

                if not in_drawdown:
                    # Start new drawdown period
                    in_drawdown = True
                    current_period = {
                        'start': peak_date,
                        'peak': running_max,
                        'trough': price,
                        'trough_date': date,
                        'max_drawdown': drawdown_magnitude,
                    }
                else:
                    # Continue drawdown - update if new trough
                    if drawdown_magnitude > current_period['max_drawdown']:
                        current_period['max_drawdown'] = drawdown_magnitude
                        current_period['trough'] = price
                        current_period['trough_date'] = date

        # Handle ongoing drawdown at end of data
        if in_drawdown and current_period is not None:
            current_period['end'] = prices.index[-1]
            current_period['duration_days'] = len(
                prices[current_period['start']:current_period['end']]
            )
            if current_period['max_drawdown'] >= min_drawdown:
                periods.append(current_period)

        return periods

    def get_all_drawdown_magnitudes(
        self,
        min_drawdown: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """Get all drawdown magnitudes for each asset.

        Args:
            min_drawdown: Minimum drawdown magnitude to include.

        Returns:
            Dictionary mapping asset names to arrays of drawdown magnitudes
            (as positive decimals, e.g., 0.15 for 15% drawdown).
        """
        if self._drawdown_periods is None:
            self.get_drawdown_periods(min_drawdown)

        magnitudes = {}
        for asset_name, periods in self._drawdown_periods.items():
            mags = [p['max_drawdown'] for p in periods]
            magnitudes[asset_name] = np.array(mags)

        return magnitudes

    def get_summary_statistics(
        self,
        min_drawdown: float = 0.01
    ) -> pd.DataFrame:
        """Calculate summary statistics for drawdowns.

        Args:
            min_drawdown: Minimum drawdown magnitude to include.

        Returns:
            DataFrame with summary statistics for each asset.
            Columns include: count, mean, std, min, max, median.
        """
        magnitudes = self.get_all_drawdown_magnitudes(min_drawdown)

        stats = {}
        for asset_name, mags in magnitudes.items():
            if len(mags) > 0:
                stats[asset_name] = {
                    'count': len(mags),
                    'mean': np.mean(mags),
                    'std': np.std(mags),
                    'min': np.min(mags),
                    'max': np.max(mags),
                    'median': np.median(mags),
                    'q25': np.percentile(mags, 25),
                    'q75': np.percentile(mags, 75),
                }
            else:
                stats[asset_name] = {
                    'count': 0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                }

        return pd.DataFrame(stats).T

    def get_worst_drawdowns(
        self,
        n: int = 5,
        min_drawdown: float = 0.01
    ) -> Dict[str, List[Dict]]:
        """Get the N worst drawdowns for each asset.

        Args:
            n: Number of worst drawdowns to return.
            min_drawdown: Minimum drawdown magnitude to include.

        Returns:
            Dictionary mapping asset names to lists of their worst drawdowns.
        """
        if self._drawdown_periods is None:
            self.get_drawdown_periods(min_drawdown)

        worst_drawdowns = {}
        for asset_name, periods in self._drawdown_periods.items():
            # Sort by max_drawdown in descending order
            sorted_periods = sorted(
                periods,
                key=lambda x: x['max_drawdown'],
                reverse=True
            )
            worst_drawdowns[asset_name] = sorted_periods[:n]

        return worst_drawdowns
