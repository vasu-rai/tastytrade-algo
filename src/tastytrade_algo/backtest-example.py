# src/tastytrade_algo/backtest-example.py

import os
# sys import might not be needed if running with -m
# import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import argparse # Import argparse for command-line arguments

# Import strategy and backtest components using relative paths
from .strategy import MovingAverageCrossover
from .backtest import BacktestEngine  # Assumes backtest-framework.py was renamed
from .data_fetcher import DataFetcher
# from .tastytrade_api import TastytradeAPI # No longer directly used here

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
load_dotenv()

def run_backtest(
    symbol='SPY', # Default equity symbol
    start_date='2020-01-01',
    end_date='2023-12-31',
    short_window=20,
    long_window=50,
    initial_capital=100000.0,
    max_capital_risk_pct=0.02,
    max_buying_power_pct=0.6,
    commission=0.0,
    slippage=0.001,
    data_dir='data',
    report_dir='reports',
    # ADDED: Arguments to control instrument type and option symbol
    instrument_type_arg = 'equity',
    option_symbol_arg = None
):
    """
    Run a backtest with the specified parameters. Can handle equity or options.
    """
    # --- Determine Symbol and Type ---
    if instrument_type_arg == 'option' and option_symbol_arg:
         # Use option symbol if type is option and symbol is provided
         symbol_to_use = option_symbol_arg
         instrument_type_to_use = 'option'
         # Use option symbol for report directory name (sanitized)
         run_label = option_symbol_arg.replace(':','_').replace('/','_')
         logger.info(f"Configuring backtest for OPTION: {symbol_to_use}")
    else:
         # Default to equity symbol
         symbol_to_use = symbol
         instrument_type_to_use = 'equity'
         run_label = symbol # Use equity symbol for report dir
         logger.info(f"Configuring backtest for EQUITY: {symbol_to_use}")
    # --- End Determine Symbol and Type ---

    logger.info(f"Starting backtest for {symbol_to_use} ({instrument_type_to_use}) from {start_date} to {end_date}")

    # --- Create Output Directories ---
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use run_label which is either equity or sanitized option symbol
    run_dir = report_path / f"run_{timestamp}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Report directory: {run_dir}")
    # --- End Create Output Directories ---

    # --- Save Parameters ---
    params = {
        'symbol_tested': symbol_to_use, # Save the actual symbol used
        'instrument_type': instrument_type_to_use,
        'start_date': start_date,
        'end_date': end_date,
        'short_window': short_window,
        'long_window': long_window,
        'initial_capital': initial_capital,
        'max_capital_risk_pct': max_capital_risk_pct,
        'max_buying_power_pct': max_buying_power_pct,
        'commission': commission,
        'slippage': slippage
        # Add any other relevant params
    }
    try:
        with open(run_dir / 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4)
    except Exception as e:
         logger.error(f"Failed to save parameters: {e}")
    # --- End Save Parameters ---


    # --- Initialize Data Fetcher with API Keys ---
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    polygon_key = os.getenv('POLYGON_API_KEY') # Get Polygon key

    # Check for keys based on intended use
    if instrument_type_to_use == 'equity' and not alpha_vantage_key:
         logger.warning("ALPHA_VANTAGE_API_KEY not found in .env file. Equity fetching might fail.")
         # Allow to continue, maybe Tastytrade works or cache exists

    if instrument_type_to_use == 'option' and not polygon_key:
        logger.error("POLYGON_API_KEY not found in .env file. Cannot fetch options data.")
        return None # Cannot proceed without key for options

    data_fetcher = DataFetcher(
        cache_dir=str(data_path / 'cache'),
        api_key=alpha_vantage_key, # For Alpha Vantage fallback
        polygon_api_key=polygon_key, # Pass Polygon key
        tastytrade_api=None # Explicitly set Tastytrade API object to None
    )
    # --- End Initialize Data Fetcher ---


    # --- Fetch Historical Data ---
    logger.info(f"Fetching historical data for {symbol_to_use}")
    data = data_fetcher.get_historical_data(
        symbol=symbol_to_use,
        start_date=start_date,
        end_date=end_date,
        interval='daily', # Adjust interval as needed ('minute', 'hourly', etc.)
        instrument_type=instrument_type_to_use # Pass the determined instrument type
    )
    # --- End Fetch Historical Data ---


    # --- Check and Prepare Data ---
    if data is None or data.empty:
        logger.error(f"No data available for {symbol_to_use} after fetching attempt. Stopping backtest.")
        return None

    logger.info(f"Preparing data for {symbol_to_use}...")
    data = data_fetcher.prepare_data(data)
    if data is None or data.empty:
        logger.error(f"Data preparation failed for {symbol_to_use}. Stopping backtest.")
        return None
    logger.info(f"Data prepared. Shape: {data.shape}")
    # --- End Check and Prepare Data ---


    # --- Initialize Strategy and Engine ---
    # IMPORTANT: Current strategy and engine are EQUITY based.
    # Running options data through them will likely produce meaningless results.
    # This needs significant modification for a proper options backtest.
    if instrument_type_to_use == 'option':
        logger.warning("Running OPTIONS data through EQUITY strategy/engine. Results are likely NOT meaningful.")
        # In a real scenario, you would initialize an options-specific strategy/engine here
        # strategy = MyOptionsStrategy(...)
        # engine = OptionsBacktestEngine(...)
    else:
        logger.info("Initializing equity strategy and engine.")

    # Using equity strategy/engine regardless for now
    strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        max_capital_risk_pct=max_capital_risk_pct,
        max_buying_power_pct=max_buying_power_pct
    )
    # --- End Initialize Strategy and Engine ---


    # --- Run Backtest ---
    logger.info(f"Running backtest simulation for {symbol_to_use}...")
    try:
        metrics = engine.run(strategy, data)
        logger.info("Backtest simulation finished.")
    except Exception as e:
        logger.error(f"Error during backtest simulation: {e}", exc_info=True)
        return None
    # --- End Run Backtest ---


    # --- Generate Report ---
    logger.info("Generating report...")
    try:
        report = engine.generate_report(save_dir=str(run_dir))
        print("\n" + "="*30 + " BACKTEST SUMMARY " + "="*30)
        print(report)
        print("="*80 + "\n")

        # Plot results (might raise errors if data is unexpected for plots)
        logger.info("Plotting results...")
        engine.plot_equity_curve(save_path=str(run_dir / 'equity_curve.png'))
        plt.close() # Close plot to prevent display issues in loops/scripts
        engine.plot_drawdown(save_path=str(run_dir / 'drawdown.png'))
        plt.close() # Close plot

        # Save signals (still equity signals)
        logger.info("Saving signals...")
        signals = strategy.generate_signals(data)
        signals.to_csv(run_dir / 'signals.csv')

        logger.info(f"Backtest completed. Results saved to {run_dir}")
    except Exception as e:
        logger.error(f"Error during report generation or plotting: {e}", exc_info=True)
        # Metrics might still be valid even if reporting fails
    # --- End Generate Report ---

    return metrics

# Note: run_parameter_optimization function is NOT updated for options
# It would need similar logic for data fetching and likely a different optimization target/strategy
def run_parameter_optimization(
    # ... existing parameters ...
):
    """ (This function still uses the old equity-only data fetching) """
    logger.warning("run_parameter_optimization is not updated for options data.")
    # ... existing implementation ...


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run backtest for equity or options.')

    # Instrument Selection
    parser.add_argument('--instrument-type', type=str, default='equity', choices=['equity', 'option'],
                        help="Type of instrument ('equity' or 'option'). Default: equity")
    parser.add_argument('--symbol', type=str, default='SPY',
                        help="Ticker symbol (Equity symbol if type is equity). Default: SPY")
    parser.add_argument('--option-symbol', type=str,
                        help="Option symbol in Polygon format (e.g., O:SPY...) if instrument-type is option.")

    # Date Range
    parser.add_argument('--start-date', type=str, default='2023-01-01', # Shorter default for faster testing
                        help='Start date (YYYY-MM-DD). Default: 2023-01-01')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD). Default: today')

    # Strategy parameters (for MovingAverageCrossover)
    parser.add_argument('--short-window', type=int, default=20, help='Short moving average window')
    parser.add_argument('--long-window', type=int, default=50, help='Long moving average window')

    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--max-risk', type=float, default=0.02, help='Maximum capital risk per trade')
    parser.add_argument('--max-buying-power', type=float, default=0.6, help='Maximum buying power to deploy')
    parser.add_argument('--commission', type=float, default=0.0, help='Commission per trade')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage as percentage of price')

    # Output parameters
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data storage')
    parser.add_argument('--report-dir', type=str, default='reports', help='Directory for report output')

    # Mode selection
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization (Equity only currently)')

    args = parser.parse_args()
    # --- End Argument Parsing ---

    # --- Execute ---
    if args.optimize:
        if args.instrument_type == 'option':
             print("ERROR: Optimization mode currently only supports equity.")
        else:
             # Run parameter optimization (Equity only)
             run_parameter_optimization(
                 symbol=args.symbol,
                 start_date=args.start_date,
                 end_date=args.end_date,
                 initial_capital=args.initial_capital,
                 output_dir=args.report_dir + '/optimization'
                 # Pass other relevant args if needed by optimization function
             )
    else:
        # Run single backtest (Equity or Option)
        run_backtest(
            # Pass arguments from argparse
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            short_window=args.short_window,
            long_window=args.long_window,
            initial_capital=args.initial_capital,
            max_capital_risk_pct=args.max_risk,
            max_buying_power_pct=args.max_buying_power,
            commission=args.commission,
            slippage=args.slippage,
            data_dir=args.data_dir,
            report_dir=args.report_dir,
            # Pass new instrument args
            instrument_type_arg=args.instrument_type,
            option_symbol_arg=args.option_symbol
        )
    # --- End Execute ---
    