"""Unit tests for data fetching module."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stock_drawdown.data_fetcher import StockDataFetcher  # noqa: E402


class TestStockDataFetcher:
    """Test cases for StockDataFetcher class."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        fetcher = StockDataFetcher()

        assert fetcher.indices == StockDataFetcher.DEFAULT_INDICES
        assert fetcher.end_date == datetime.now().strftime("%Y-%m-%d")

        # Check start date is approximately 10 years ago
        expected_start = datetime.now() - timedelta(days=365 * 10)
        actual_start = datetime.strptime(fetcher.start_date, "%Y-%m-%d")
        assert abs((actual_start - expected_start).days) <= 1

    def test_initialization_with_custom_dates(self):
        """Test initialization with custom dates."""
        start = "2020-01-01"
        end = "2023-12-31"

        fetcher = StockDataFetcher(start_date=start, end_date=end)

        assert fetcher.start_date == start
        assert fetcher.end_date == end

    def test_initialization_with_custom_indices(self):
        """Test initialization with custom indices."""
        custom_indices = {"TEST": "^TEST"}
        fetcher = StockDataFetcher(indices=custom_indices)

        assert fetcher.indices == custom_indices

    def test_invalid_date_format(self):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            StockDataFetcher(start_date="2020/01/01")

        with pytest.raises(ValueError, match="Invalid date format"):
            StockDataFetcher(end_date="invalid-date")

    def test_start_date_after_end_date(self):
        """Test that start_date after end_date raises ValueError."""
        with pytest.raises(ValueError, match="must be before"):
            StockDataFetcher(
                start_date="2023-12-31",
                end_date="2020-01-01"
            )

    def test_start_date_equals_end_date(self):
        """Test that start_date equal to end_date raises ValueError."""
        with pytest.raises(ValueError, match="must be before"):
            StockDataFetcher(
                start_date="2023-01-01",
                end_date="2023-01-01"
            )

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_fetch_single_index_success(self, mock_download):
        """Test successful fetching of a single index."""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(105, 115, 10),
            'Low': np.linspace(95, 105, 10),
            'Close': np.linspace(100, 110, 10),
            'Volume': [1000000] * 10
        }, index=dates)

        mock_download.return_value = mock_data

        fetcher = StockDataFetcher(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        result = fetcher._fetch_single_index("^GSPC", "SP500")

        assert result is not None
        assert len(result) == 10
        assert 'Close' in result.columns
        assert not result.isnull().any().any()

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_fetch_single_index_empty_data(self, mock_download):
        """Test handling of empty data from API."""
        mock_download.return_value = pd.DataFrame()

        fetcher = StockDataFetcher(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        with pytest.raises(ValueError, match="No data returned"):
            fetcher._fetch_single_index("^INVALID", "INVALID")

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_fetch_data_with_invalid_index_name(self, mock_download):
        """Test fetching with invalid index name."""
        fetcher = StockDataFetcher()

        with pytest.raises(ValueError, match="Unknown index names"):
            fetcher.fetch_data(["INVALID_INDEX"])

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_clean_data_with_missing_values(self, mock_download):
        """Test data cleaning with missing values."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(105, 115, 10),
            'Low': np.linspace(95, 105, 10),
            'Close': np.linspace(100, 110, 10),
            'Volume': [1000000] * 10
        }, index=dates)

        # Introduce some NaN values
        mock_data.loc[dates[2], 'Close'] = np.nan
        mock_data.loc[dates[5], 'Open'] = np.nan

        mock_download.return_value = mock_data

        fetcher = StockDataFetcher(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        result = fetcher._fetch_single_index("^GSPC", "SP500")

        # Should have filled NaN values
        assert not result.isnull().any().any()

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_clean_data_with_too_many_missing_values(self, mock_download):
        """Test that too many missing values raises error."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Close': [100] + [np.nan] * 9
        }, index=dates)

        mock_download.return_value = mock_data

        fetcher = StockDataFetcher(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        with pytest.raises(ValueError, match="Too many missing values"):
            fetcher._fetch_single_index("^GSPC", "SP500")

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_clean_data_with_non_positive_prices(self, mock_download):
        """Test that non-positive prices raise error."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Close': [100, 90, 0, 70, 60, 50, 40, 30, 20, 10]
        }, index=dates)

        mock_download.return_value = mock_data

        fetcher = StockDataFetcher(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        with pytest.raises(ValueError, match="non-positive prices"):
            fetcher._fetch_single_index("^GSPC", "SP500")

    @patch('stock_drawdown.data_fetcher.yf.download')
    def test_get_close_prices(self, mock_download):
        """Test extraction of close prices."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.linspace(100, 110, 10),
            'High': np.linspace(105, 115, 10),
            'Low': np.linspace(95, 105, 10),
            'Close': np.linspace(100, 110, 10),
            'Volume': [1000000] * 10
        }, index=dates)

        mock_download.return_value = mock_data

        fetcher = StockDataFetcher(
            indices={"INDEX1": "^IDX1", "INDEX2": "^IDX2"},
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        data = fetcher.fetch_data()
        close_prices = fetcher.get_close_prices(data)

        assert isinstance(close_prices, pd.DataFrame)
        assert 'INDEX1' in close_prices.columns
        assert 'INDEX2' in close_prices.columns
        assert len(close_prices) == 10

    def test_default_indices_coverage(self):
        """Test that default indices include major markets."""
        indices = StockDataFetcher.DEFAULT_INDICES

        # Check European indices
        assert 'STOXX50' in indices
        assert 'FTSE100' in indices
        assert 'DAX' in indices
        assert 'CAC40' in indices

        # Check Japan
        assert 'NIKKEI225' in indices

        # Check USA
        assert 'SP500' in indices
        assert 'DJIA' in indices
        assert 'NASDAQ' in indices
