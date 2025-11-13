"""Main script for stock market drawdown analysis.

This script performs a comprehensive analysis of drawdowns in major stock
market indices, including:
1. Fetching historical data
2. Computing drawdowns
3. Fitting Pareto distributions
4. Creating visualizations

Usage:
    python main.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stock_drawdown import (  # noqa: E402
    StockDataFetcher,
    DrawdownCalculator,
    ParetoDistributionFitter,
    DrawdownVisualizer,
)


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Analyze drawdowns in stock market indices'
    )

    default_end = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format '
             '(default: 1900-01-01, fetches all available data)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help=f'End date in YYYY-MM-DD format (default: today - {default_end})'
    )

    parser.add_argument(
        '--min-drawdown',
        type=float,
        default=0.01,
        help='Minimum drawdown magnitude to analyze (default: 0.01 = 1%%)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )

    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    print("=" * 70)
    print("Stock Market Drawdown Analysis")
    print("=" * 70)
    print()

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Minimum Drawdown: {args.min_drawdown * 100:.1f}%")
    print(f"Output Directory: {output_dir}")
    print()

    # Step 1: Fetch data
    print("-" * 70)
    print("STEP 1: Fetching Stock Index Data")
    print("-" * 70)

    try:
        fetcher = StockDataFetcher(
            start_date=args.start_date,
            end_date=args.end_date
        )

        print(f"Analysis Period: {fetcher.start_date} to {fetcher.end_date}")

        # Fetch data for main indices
        indices_to_fetch = [
            'STOXX50', 'FTSE100', 'DAX', 'CAC40',  # Europe
            'NIKKEI225',                            # Japan
            'SP500', 'DJIA', 'NASDAQ'               # USA
        ]

        print(f"\nFetching data for {len(indices_to_fetch)} indices...")
        data = fetcher.fetch_data(indices_to_fetch)

        print(f"\nSuccessfully fetched data for {len(data)} indices:")
        for name in data.keys():
            print(f"  - {name}: {len(data[name])} trading days")

        # Extract close prices
        close_prices = fetcher.get_close_prices(data)

    except Exception as e:
        print(f"\nError fetching data: {e}")
        return 1

    # Step 2: Calculate drawdowns
    print("\n" + "-" * 70)
    print("STEP 2: Computing Drawdowns")
    print("-" * 70)

    try:
        calculator = DrawdownCalculator(close_prices)

        # Calculate drawdown series
        print("\nCalculating drawdown series...")
        drawdowns = calculator.calculate_drawdowns()

        # Get drawdown magnitudes
        print(f"Identifying drawdown periods (min: {args.min_drawdown * 100:.1f}%)...")
        magnitudes = calculator.get_all_drawdown_magnitudes(
            min_drawdown=args.min_drawdown
        )

        # Print summary statistics
        print("\nDrawdown Summary Statistics:")
        summary_stats = calculator.get_summary_statistics(
            min_drawdown=args.min_drawdown
        )
        print(summary_stats.to_string())

        # Print worst drawdowns
        print("\n\nTop 3 Worst Drawdowns by Index:")
        worst_drawdowns = calculator.get_worst_drawdowns(n=3)
        for asset, dd_list in worst_drawdowns.items():
            print(f"\n{asset}:")
            for i, dd in enumerate(dd_list, 1):
                print(
                    f"  {i}. {dd['max_drawdown']*100:.2f}% "
                    f"({dd['trough_date'].strftime('%Y-%m-%d')})"
                )

    except Exception as e:
        print(f"\nError calculating drawdowns: {e}")
        return 1

    # Step 3: Fit Pareto distributions
    print("\n" + "-" * 70)
    print("STEP 3: Fitting Pareto Distributions")
    print("-" * 70)

    try:
        fitter = ParetoDistributionFitter(magnitudes)

        # Fit all assets
        print("\nFitting Pareto distributions using MLE...")
        fitted_params = fitter.fit_all_assets(method='mle')

        print("\nFitted Parameters (shape, loc, scale):")
        for asset, params in fitted_params.items():
            shape, loc, scale = params
            if not pd.isna(shape):
                print(
                    f"  {asset:12s}: α={shape:.3f}, "
                    f"x_m={scale*100:.2f}%"
                )

        # Calculate goodness of fit
        print("\nCalculating goodness-of-fit statistics...")
        gof_stats = fitter.calculate_all_goodness_of_fit()

        print("\nGoodness-of-Fit Statistics:")
        print(f"{'Index':<12} {'KS Statistic':<15} {'p-value':<15} {'Log-Likelihood'}")
        print("-" * 60)
        for asset, stats in gof_stats.items():
            print(
                f"{asset:<12} "
                f"{stats['ks_statistic']:<15.4f} "
                f"{stats['ks_pvalue']:<15.4f} "
                f"{stats['log_likelihood']:.2f}"
            )

        # Calculate Value at Risk
        print("\n\nValue at Risk (95% confidence):")
        for asset in magnitudes.keys():
            try:
                var_95 = fitter.calculate_value_at_risk(asset, 0.95)
                print(f"  {asset:12s}: {var_95*100:.2f}%")
            except Exception:
                pass

    except Exception as e:
        print(f"\nError fitting distributions: {e}")
        return 1

    # Step 4: Create visualizations
    print("\n" + "-" * 70)
    print("STEP 4: Creating Visualizations")
    print("-" * 70)

    try:
        visualizer = DrawdownVisualizer()

        # Plot drawdown series
        print("\nCreating drawdown time series plot...")
        visualizer.plot_drawdown_series(
            drawdowns,
            save_path=str(output_dir / 'drawdown_series.png')
        )

        # Plot histograms
        print("Creating drawdown histograms...")
        visualizer.plot_drawdown_histogram(
            magnitudes,
            save_path=str(output_dir / 'drawdown_histograms.png')
        )

        # Plot Pareto fits
        print("Creating Pareto distribution fits...")
        visualizer.plot_pareto_fit(
            magnitudes,
            fitted_params,
            save_path=str(output_dir / 'pareto_fits.png')
        )

        # Plot log-log plots
        print("Creating log-log plots...")
        visualizer.plot_pareto_fit_log(
            magnitudes,
            fitted_params,
            save_path=str(output_dir / 'pareto_fits_loglog.png')
        )

        # Plot QQ plots
        print("Creating QQ plots...")
        visualizer.plot_qq(
            magnitudes,
            fitted_params,
            save_path=str(output_dir / 'qq_plots.png')
        )

        # Create comprehensive summary report
        print("Creating comprehensive summary report...")
        visualizer.create_summary_report(
            magnitudes,
            fitted_params,
            gof_stats,
            save_path=str(output_dir / 'summary_report.png')
        )

        print(f"\nAll visualizations saved to: {output_dir}/")

    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save numerical results
    print("\n" + "-" * 70)
    print("STEP 5: Saving Numerical Results")
    print("-" * 70)

    try:
        # Save summary statistics
        summary_stats.to_csv(output_dir / 'summary_statistics.csv')
        print(f"Saved summary statistics to: {output_dir}/summary_statistics.csv")

        # Save fitted parameters
        params_df = pd.DataFrame(
            fitted_params,
            index=['shape', 'loc', 'scale']
        ).T
        params_df.to_csv(output_dir / 'fitted_parameters.csv')
        print(f"Saved fitted parameters to: {output_dir}/fitted_parameters.csv")

        # Save goodness of fit
        gof_df = pd.DataFrame(gof_stats).T
        gof_df.to_csv(output_dir / 'goodness_of_fit.csv')
        print(f"Saved goodness-of-fit to: {output_dir}/goodness_of_fit.csv")

    except Exception as e:
        print(f"\nError saving results: {e}")
        return 1

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    import pandas as pd
    sys.exit(main())
