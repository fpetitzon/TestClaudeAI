"""Module for fetching stock market index data."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


class StockDataFetcher:
    """Fetch historical stock market index data.

    This class provides functionality to download historical price data
    for major stock market indices from Yahoo Finance.

    Attributes:
        DEFAULT_INDICES: Dictionary of default market indices with their
            ticker symbols.
    """

    DEFAULT_INDICES = {
        # European indices
        "STOXX50": "^STOXX50E",  # Euro Stoxx 50
        "FTSE100": "^FTSE",      # UK FTSE 100
        "DAX": "^GDAXI",         # German DAX
        "CAC40": "^FCHI",        # French CAC 40
        # Japan
        "NIKKEI225": "^N225",    # Nikkei 225
        "TOPIX": "^TOPX",        # Tokyo Stock Price Index
        # USA
        "SP500": "^GSPC",        # S&P 500
        "DJIA": "^DJI",          # Dow Jones Industrial Average
        "NASDAQ": "^IXIC",       # NASDAQ Composite
    }

    def __init__(
        self,
        indices: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """Initialize the StockDataFetcher.

        Args:
            indices: Dictionary mapping index names to ticker symbols.
                If None, uses DEFAULT_INDICES.
            start_date: Start date in 'YYYY-MM-DD' format.
                If None, defaults to 1900-01-01 (fetches all available data).
            end_date: End date in 'YYYY-MM-DD' format.
                If None, defaults to today.

        Raises:
            ValueError: If date format is invalid or start_date > end_date.
        """
        self.indices = indices if indices is not None else self.DEFAULT_INDICES

        # Set default dates
        if end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            self.end_date = end_date

        if start_date is None:
            # Default to 1900-01-01 to get all available historical data
            self.start_date = "1900-01-01"
        else:
            self.start_date = start_date

        # Validate dates
        self._validate_dates()

    def _validate_dates(self) -> None:
        """Validate date formats and logical ordering.

        Raises:
            ValueError: If date format is invalid or start_date > end_date.
        """
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}"
            ) from e

        if start >= end:
            raise ValueError(
                f"start_date ({self.start_date}) must be before "
                f"end_date ({self.end_date})"
            )

    def fetch_data(
        self,
        index_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical price data for specified indices.

        Args:
            index_names: List of index names to fetch. If None, fetches all
                indices defined in self.indices.

        Returns:
            Dictionary mapping index names to DataFrames with OHLCV data.
            Each DataFrame contains columns: Open, High, Low, Close, Volume.

        Raises:
            ValueError: If an index name is not found in self.indices.
            RuntimeError: If data fetching fails for any index.
        """
        if index_names is None:
            index_names = list(self.indices.keys())

        # Validate index names
        invalid_indices = set(index_names) - set(self.indices.keys())
        if invalid_indices:
            raise ValueError(
                f"Unknown index names: {invalid_indices}. "
                f"Available indices: {list(self.indices.keys())}"
            )

        data = {}
        failed_indices = []

        for name in index_names:
            ticker = self.indices[name]
            try:
                df = self._fetch_single_index(ticker, name)
                if df is not None and not df.empty:
                    data[name] = df
                else:
                    failed_indices.append(name)
            except Exception as e:
                print(f"Warning: Failed to fetch data for {name}: {e}")
                failed_indices.append(name)

        if not data:
            raise RuntimeError(
                f"Failed to fetch data for all indices. "
                f"Failed: {failed_indices}"
            )

        if failed_indices:
            print(
                f"Warning: Failed to fetch data for: "
                f"{', '.join(failed_indices)}"
            )

        return data

    def _fetch_single_index(
        self,
        ticker: str,
        name: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single index.

        Args:
            ticker: Yahoo Finance ticker symbol.
            name: Human-readable index name.

        Returns:
            DataFrame with OHLCV data, or None if fetch failed.

        Raises:
            Exception: If download fails or data is invalid.
        """
        print(f"Fetching data for {name} ({ticker})...")

        # Use pandas Timestamp for better yfinance compatibility
        start_ts = pd.Timestamp(self.start_date)
        end_ts = pd.Timestamp(self.end_date)

        # Download data
        # Note: yfinance can be picky about date formats
        # Using pandas Timestamps ensures compatibility
        # If start_date/end_date are strings like '2024-01-01'
        start_ts = pd.Timestamp(datetime.strptime(self.start_date, '%Y-%m-%d'))
        end_ts = pd.Timestamp(datetime.strptime(self.end_date, '%Y-%m-%d'))
        print( "Using start and end timestamps: ")
        print(f"  start_ts: {start_ts}")
        print(f"  end_ts:   {end_ts}")


        data = yf.download(
            ticker,
            start=start_ts,
            end=end_ts,
            progress=False
        )

        print(f"Downloaded data for {name}:")
        print(data.head())

        if data.empty:
            raise ValueError(f"No data returned for {name} ({ticker})")

        # Clean the data
        data = self._clean_data(data, name)

        print(f"Successfully fetched {len(data)} records for {name}")
        return data

    def _clean_data(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Clean and validate the fetched data.

        Args:
            df: Raw DataFrame from yfinance.
            name: Index name for error messages.

        Returns:
            Cleaned DataFrame with NaN values handled.

        Raises:
            ValueError: If data is invalid or has too many missing values.
        """
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        if missing_pct.max() > 10:
            raise ValueError(
                f"{name}: Too many missing values. "
                f"Max missing: {missing_pct.max():.2f}%"
            )

        # Forward fill then backward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def get_close_prices(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Extract close prices from fetched data.

        Args:
            data: Dictionary of DataFrames from fetch_data().

        Returns:
            DataFrame with close prices for each index.
            Columns are index names, index is the date.
        """
        if not data:
            raise ValueError("No data provided to extract close prices")

        # Extract close prices as Series and combine
        close_series = {}
        for name, df in data.items():
            if 'Close' in df.columns:
                close_series[name] = df['Close'].copy()
            else:
                raise ValueError(
                    f"'Close' column not found in data for {name}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Combine all series into a DataFrame using concat for proper alignment
        result = pd.concat(close_series, axis=1)
        result.columns = list(close_series.keys())

        return result.sort_index()
