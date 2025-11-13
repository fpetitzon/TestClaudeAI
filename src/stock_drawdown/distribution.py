"""Module for fitting Pareto distributions to drawdown data."""

from typing import Dict, Tuple

import numpy as np
from scipy import stats


class ParetoDistributionFitter:
    """Fit Pareto distributions to drawdown magnitude data.

    The Pareto distribution is often used to model extreme events
    in financial markets, including large drawdowns. This class provides
    methods to fit the distribution and assess goodness of fit.
    """

    def __init__(self, drawdown_magnitudes: Dict[str, np.ndarray]):
        """Initialize the ParetoDistributionFitter.

        Args:
            drawdown_magnitudes: Dictionary mapping asset names to arrays
                of drawdown magnitudes (positive decimals).

        Raises:
            ValueError: If input data is invalid.
        """
        if not drawdown_magnitudes:
            raise ValueError("drawdown_magnitudes cannot be empty")

        for asset, mags in drawdown_magnitudes.items():
            if len(mags) == 0:
                raise ValueError(f"No drawdowns for asset: {asset}")
            if (mags <= 0).any():
                raise ValueError(
                    f"Drawdown magnitudes must be positive for {asset}"
                )

        self.drawdown_magnitudes = drawdown_magnitudes
        self._fitted_params = {}
        self._goodness_of_fit = {}

    def fit_pareto(
        self,
        asset_name: str,
        method: str = 'mle'
    ) -> Tuple[float, float, float]:
        """Fit a Pareto distribution to drawdown data for one asset.

        The Pareto distribution (Type I) has PDF:
            f(x) = (alpha * x_m^alpha) / x^(alpha + 1)
        where x >= x_m, alpha > 0 (shape), and x_m > 0 (scale/minimum).

        Args:
            asset_name: Name of the asset to fit.
            method: Fitting method. Options:
                - 'mle': Maximum Likelihood Estimation (default)
                - 'moments': Method of moments

        Returns:
            Tuple of (shape, loc, scale) parameters.
            For Pareto: shape=alpha, loc=0 (fixed), scale=x_m

        Raises:
            ValueError: If asset_name not found or method is invalid.
        """
        if asset_name not in self.drawdown_magnitudes:
            raise ValueError(f"Asset not found: {asset_name}")

        if method not in ['mle', 'moments']:
            raise ValueError(f"Invalid method: {method}. Use 'mle' or 'moments'")

        data = self.drawdown_magnitudes[asset_name]

        if method == 'mle':
            # Use scipy's built-in MLE fitting
            # Note: scipy.stats.pareto uses shape parameter b and scale parameter
            # where the distribution is: f(x) = b * scale^b / x^(b+1)
            shape, loc, scale = stats.pareto.fit(data, floc=0)
        else:  # moments
            shape, loc, scale = self._fit_pareto_moments(data)

        # Store the fitted parameters
        self._fitted_params[asset_name] = (shape, loc, scale)

        return shape, loc, scale

    def _fit_pareto_moments(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Fit Pareto distribution using method of moments.

        Args:
            data: Array of drawdown magnitudes.

        Returns:
            Tuple of (shape, loc, scale) parameters.
        """
        # For Pareto distribution:
        # Mean = x_m * alpha / (alpha - 1) for alpha > 1
        # Variance = x_m^2 * alpha / ((alpha - 1)^2 * (alpha - 2)) for alpha > 2

        x_m = np.min(data)  # Scale parameter (minimum value)
        mean = np.mean(data)

        # From mean formula: alpha = mean / (mean - x_m)
        # This requires mean > x_m
        if mean <= x_m:
            # Fallback to MLE if moments method fails
            return stats.pareto.fit(data, floc=0)

        alpha = mean / (mean - x_m)

        # Ensure alpha is positive and reasonable
        if alpha <= 1:
            alpha = 1.5  # Minimum reasonable value

        return alpha, 0, x_m

    def fit_all_assets(
        self, method: str = 'mle'
    ) -> Dict[str, Tuple[float, float, float]]:
        """Fit Pareto distributions to all assets.

        Args:
            method: Fitting method ('mle' or 'moments').

        Returns:
            Dictionary mapping asset names to fitted parameters.
        """
        params = {}
        for asset_name in self.drawdown_magnitudes.keys():
            try:
                params[asset_name] = self.fit_pareto(asset_name, method)
            except Exception as e:
                print(f"Warning: Failed to fit {asset_name}: {e}")
                params[asset_name] = (np.nan, np.nan, np.nan)

        return params

    def calculate_goodness_of_fit(
        self,
        asset_name: str
    ) -> Dict[str, float]:
        """Calculate goodness-of-fit statistics for fitted distribution.

        Args:
            asset_name: Name of the asset.

        Returns:
            Dictionary with goodness-of-fit metrics:
                - ks_statistic: Kolmogorov-Smirnov statistic
                - ks_pvalue: P-value for KS test
                - ad_statistic: Anderson-Darling statistic
                - log_likelihood: Log-likelihood of the fit

        Raises:
            RuntimeError: If Pareto has not been fitted for this asset.
        """
        if asset_name not in self._fitted_params:
            raise RuntimeError(
                f"Pareto not fitted for {asset_name}. Call fit_pareto first."
            )

        data = self.drawdown_magnitudes[asset_name]
        shape, loc, scale = self._fitted_params[asset_name]

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(
            data,
            lambda x: stats.pareto.cdf(x, shape, loc, scale)
        )

        # Anderson-Darling test (approximate for Pareto)
        # Transform to standard uniform under the fitted distribution
        uniform_data = stats.pareto.cdf(data, shape, loc, scale)
        ad_result = stats.anderson(uniform_data, dist='uniform')
        ad_stat = ad_result.statistic

        # Log-likelihood
        log_likelihood = np.sum(stats.pareto.logpdf(data, shape, loc, scale))

        results = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'ad_statistic': ad_stat,
            'log_likelihood': log_likelihood,
        }

        self._goodness_of_fit[asset_name] = results
        return results

    def calculate_all_goodness_of_fit(self) -> Dict[str, Dict[str, float]]:
        """Calculate goodness-of-fit for all fitted distributions.

        Returns:
            Dictionary mapping asset names to goodness-of-fit metrics.

        Raises:
            RuntimeError: If no distributions have been fitted yet.
        """
        if not self._fitted_params:
            raise RuntimeError(
                "No fitted distributions. Call fit_all_assets first."
            )

        results = {}
        for asset_name in self._fitted_params.keys():
            if not np.isnan(self._fitted_params[asset_name][0]):
                try:
                    results[asset_name] = self.calculate_goodness_of_fit(
                        asset_name
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to calculate GOF for {asset_name}: {e}"
                    )
                    results[asset_name] = {
                        'ks_statistic': np.nan,
                        'ks_pvalue': np.nan,
                        'ad_statistic': np.nan,
                        'log_likelihood': np.nan,
                    }

        return results

    def get_fitted_params(self, asset_name: str) -> Tuple[float, float, float]:
        """Get fitted parameters for an asset.

        Args:
            asset_name: Name of the asset.

        Returns:
            Tuple of (shape, loc, scale) parameters.

        Raises:
            RuntimeError: If Pareto has not been fitted for this asset.
        """
        if asset_name not in self._fitted_params:
            raise RuntimeError(
                f"Pareto not fitted for {asset_name}. Call fit_pareto first."
            )

        return self._fitted_params[asset_name]

    def get_all_fitted_params(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all fitted parameters.

        Returns:
            Dictionary mapping asset names to fitted parameters.
        """
        return self._fitted_params.copy()

    def calculate_tail_probability(
        self,
        asset_name: str,
        threshold: float
    ) -> float:
        """Calculate probability of drawdown exceeding threshold.

        Args:
            asset_name: Name of the asset.
            threshold: Drawdown magnitude threshold (positive decimal).

        Returns:
            Probability P(X > threshold) under fitted Pareto distribution.

        Raises:
            RuntimeError: If Pareto has not been fitted for this asset.
            ValueError: If threshold is not positive.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        shape, loc, scale = self.get_fitted_params(asset_name)

        # Survival function (1 - CDF)
        prob = stats.pareto.sf(threshold, shape, loc, scale)

        return prob

    def calculate_value_at_risk(
        self,
        asset_name: str,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk for drawdowns.

        VaR is the quantile of the distribution at the specified confidence level.

        Args:
            asset_name: Name of the asset.
            confidence: Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            Drawdown magnitude at the specified confidence level.

        Raises:
            RuntimeError: If Pareto has not been fitted for this asset.
            ValueError: If confidence is not in (0, 1).
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")

        shape, loc, scale = self.get_fitted_params(asset_name)

        # Calculate quantile
        var = stats.pareto.ppf(confidence, shape, loc, scale)

        return var
