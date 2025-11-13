"""Stock drawdown analysis package."""

__version__ = "1.0.0"
__author__ = "Expert Python Developer"

from .data_fetcher import StockDataFetcher
from .drawdown import DrawdownCalculator
from .distribution import ParetoDistributionFitter
from .visualization import DrawdownVisualizer

__all__ = [
    "StockDataFetcher",
    "DrawdownCalculator",
    "ParetoDistributionFitter",
    "DrawdownVisualizer",
]
