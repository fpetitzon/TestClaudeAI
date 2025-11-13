# Stock Market Drawdown Analysis

A comprehensive Python application for analyzing drawdowns in major stock market indices and fitting Pareto distributions to drawdown magnitudes.

## Features

- **Data Fetching**: Automatically fetch historical data for major stock indices from Yahoo Finance
  - European indices: STOXX 50, FTSE 100, DAX, CAC 40
  - Japanese indices: Nikkei 225, TOPIX
  - US indices: S&P 500, Dow Jones, NASDAQ

- **Drawdown Calculation**: Compute drawdowns with precision
  - Calculate drawdown time series
  - Identify individual drawdown periods
  - Analyze drawdown severity, duration, and recovery times

- **Statistical Analysis**: Fit Pareto distributions to drawdown magnitudes
  - Maximum Likelihood Estimation (MLE) and Method of Moments
  - Goodness-of-fit testing (Kolmogorov-Smirnov, Anderson-Darling)
  - Value at Risk (VaR) calculations
  - Tail probability estimation

- **Visualization**: Create publication-quality plots
  - Drawdown time series
  - Histograms with fitted distributions
  - Log-log plots for assessing tail behavior
  - QQ plots for goodness-of-fit assessment
  - Comprehensive summary reports

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd TestClaudeAI
```

2. (Optional but recommended) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the analysis with default settings (last 10 years of data):

```bash
python main.py
```

### Custom Date Range

Specify a custom date range:

```bash
python main.py --start-date 2015-01-01 --end-date 2023-12-31
```

### Custom Parameters

```bash
python main.py \
    --start-date 2015-01-01 \
    --end-date 2023-12-31 \
    --min-drawdown 0.05 \
    --output-dir results
```

### Command-Line Options

- `--start-date`: Start date in YYYY-MM-DD format (default: 10 years ago)
- `--end-date`: End date in YYYY-MM-DD format (default: today)
- `--min-drawdown`: Minimum drawdown magnitude to analyze, as decimal (default: 0.01 = 1%)
- `--output-dir`: Directory to save output files (default: output)

## Output

The analysis generates several files in the output directory:

### Visualizations (PNG)
- `drawdown_series.png`: Time series of drawdowns for all indices
- `drawdown_histograms.png`: Histograms of drawdown magnitudes
- `pareto_fits.png`: Histograms with fitted Pareto distributions
- `pareto_fits_loglog.png`: Log-log plots for tail behavior assessment
- `qq_plots.png`: QQ plots for goodness-of-fit assessment
- `summary_report.png`: Comprehensive multi-panel summary

### Data (CSV)
- `summary_statistics.csv`: Summary statistics for each index
- `fitted_parameters.csv`: Fitted Pareto distribution parameters
- `goodness_of_fit.csv`: Goodness-of-fit statistics

## Project Structure

```
TestClaudeAI/
├── src/
│   └── stock_drawdown/
│       ├── __init__.py
│       ├── data_fetcher.py      # Data fetching from Yahoo Finance
│       ├── drawdown.py          # Drawdown calculations
│       ├── distribution.py      # Pareto distribution fitting
│       └── visualization.py     # Plotting and visualization
├── tests/
│   ├── __init__.py
│   ├── test_data_fetcher.py
│   ├── test_drawdown.py
│   ├── test_distribution.py
│   └── test_visualization.py
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .gitignore
└── README.md
```

## Development

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/stock_drawdown --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_drawdown.py
```

Run tests in verbose mode:
```bash
pytest -v
```

### Code Quality

Check PEP-8 compliance:
```bash
flake8 src/ tests/ main.py
```

Format code with Black:
```bash
black src/ tests/ main.py
```

## API Documentation

### StockDataFetcher

Fetch historical stock market index data.

```python
from stock_drawdown import StockDataFetcher

fetcher = StockDataFetcher(
    start_date="2020-01-01",
    end_date="2023-12-31"
)
data = fetcher.fetch_data(['SP500', 'NIKKEI225'])
close_prices = fetcher.get_close_prices(data)
```

### DrawdownCalculator

Calculate drawdowns from price data.

```python
from stock_drawdown import DrawdownCalculator

calculator = DrawdownCalculator(close_prices)
drawdowns = calculator.calculate_drawdowns()
periods = calculator.get_drawdown_periods(min_drawdown=0.01)
magnitudes = calculator.get_all_drawdown_magnitudes()
stats = calculator.get_summary_statistics()
```

### ParetoDistributionFitter

Fit Pareto distributions to drawdown magnitudes.

```python
from stock_drawdown import ParetoDistributionFitter

fitter = ParetoDistributionFitter(magnitudes)
params = fitter.fit_all_assets(method='mle')
gof_stats = fitter.calculate_all_goodness_of_fit()
var_95 = fitter.calculate_value_at_risk('SP500', 0.95)
```

### DrawdownVisualizer

Create visualizations.

```python
from stock_drawdown import DrawdownVisualizer

visualizer = DrawdownVisualizer()
visualizer.plot_drawdown_series(drawdowns, save_path='drawdowns.png')
visualizer.plot_pareto_fit(magnitudes, params, save_path='fits.png')
visualizer.create_summary_report(
    magnitudes,
    params,
    gof_stats,
    save_path='report.png'
)
```

## Methodology

### Drawdown Definition

A drawdown at time t is defined as:

```
DD(t) = (Price(t) - RunningMax(t)) / RunningMax(t)
```

where RunningMax(t) is the maximum price observed up to time t.

### Pareto Distribution

The Pareto Type I distribution is used to model the tail behavior of drawdown magnitudes:

```
f(x) = (α * x_m^α) / x^(α + 1)
```

where:
- α > 0 is the shape parameter
- x_m > 0 is the scale parameter (minimum value)
- x ≥ x_m

### Fitting Methods

1. **Maximum Likelihood Estimation (MLE)**: Default method, provides optimal parameter estimates
2. **Method of Moments**: Alternative method based on sample moments

### Goodness-of-Fit Tests

- **Kolmogorov-Smirnov Test**: Tests whether the data follows the fitted distribution
- **Anderson-Darling Test**: More sensitive to deviations in the tails
- **Log-likelihood**: Measure of how well the distribution fits the data

## Dependencies

- `yfinance`: Fetch financial data from Yahoo Finance
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scipy`: Statistical functions and tests
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `pytest`: Testing framework
- `flake8`: PEP-8 compliance checking
- `black`: Code formatting

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`pytest`)
2. Code follows PEP-8 (`flake8`)
3. New features include tests
4. Documentation is updated

## License

MIT License

## Author

Expert Python Developer

## References

1. Johansen, A., & Sornette, D. (2001). "Large Stock Market Price Drawdowns Are Outliers"
2. Taleb, N. N. (2007). "The Black Swan: The Impact of the Highly Improbable"
3. Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). "Power-law distributions in empirical data"

## Acknowledgments

This project uses data from Yahoo Finance via the yfinance library.
