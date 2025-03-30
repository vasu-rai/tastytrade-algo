# examples/run_backtest.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy and backtest components
from src.tastytrade_algo.strategy import MovingAverageCrossover
from src.tastytrade_algo.backtest import BacktestEngine
from src.tastytrade_algo.data_fetcher import DataFetcher
from src.tastytrade_algo.tastytrade_api import TastytradeAPI  # Optional, for live data

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

# Load environment variables
load_dotenv()

def run_backtest(
    symbol='SPY',
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
    report_dir='reports'
):
    """
    Run a backtest with the specified parameters.
    
    Args:
        symbol: Ticker symbol to backtest
        start_date: Start date for backtest data
        end_date: End date for backtest data
        short_window: Short moving average window
        long_window: Long moving average window
        initial_capital: Starting capital
        max_capital_risk_pct: Maximum capital risk per trade
        max_buying_power_pct: Maximum buying power to deploy
        commission: Commission per trade
        slippage: Slippage as percentage of price
        data_dir: Directory for data storage
        report_dir: Directory for report output
    """
    logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
    
    # Create output directories
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = report_path / f"run_{timestamp}_{symbol}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'short_window': short_window,
        'long_window': long_window,
        'initial_capital': initial_capital,
        'max_capital_risk_pct': max_capital_risk_pct,
        'max_buying_power_pct': max_buying_power_pct,
        'commission': commission,
        'slippage': slippage
    }
    
    with open(run_dir / 'parameters.txt', 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(cache_dir=str(data_path / 'cache'))
    
    # Get historical data
    logger.info(f"Fetching historical data for {symbol}")
    data = data_fetcher.get_historical_data(symbol, start_date, end_date)
    
    if len(data) == 0:
        logger.error(f"No data available for {symbol}")
        return None
    
    # Prepare data for backtesting
    data = data_fetcher.prepare_data(data)
    
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        max_capital_risk_pct=max_capital_risk_pct,
        max_buying_power_pct=max_buying_power_pct
    )
    
    # Run backtest
    logger.info("Running backtest...")
    metrics = engine.run(strategy, data)
    
    # Generate report
    logger.info("Generating report...")
    report = engine.generate_report(save_dir=str(run_dir))
    
    print(report)
    
    # Plot results
    engine.plot_equity_curve(save_path=str(run_dir / 'equity_curve.png'))
    engine.plot_drawdown(save_path=str(run_dir / 'drawdown.png'))
    
    # Save DataFrame with signals for analysis
    signals = strategy.generate_signals(data)
    signals.to_csv(run_dir / 'signals.csv')
    
    logger.info(f"Backtest completed. Results saved to {run_dir}")
    
    return metrics

def run_parameter_optimization(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2023-12-31',
    short_windows=range(10, 100, 10),
    long_windows=range(20, 210, 10),
    initial_capital=100000.0,
    output_dir='optimization'
):
    """
    Run parameter optimization for the moving average strategy.
    
    Args:
        symbol: Ticker symbol
        start_date: Start date
        end_date: End date
        short_windows: Range of short MA values to test
        long_windows: Range of long MA values to test
        initial_capital: Starting capital
        output_dir: Directory for optimization results
    """
    logger.info(f"Starting parameter optimization for {symbol}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(cache_dir='data/cache')
    
    # Get historical data
    data = data_fetcher.get_historical_data(symbol, start_date, end_date)
    data = data_fetcher.prepare_data(data)
    
    if len(data) == 0:
        logger.error(f"No data available for {symbol}")
        return None
    
    # Store results
    results = []
    
    # Loop through parameter combinations
    for short_window in short_windows:
        for long_window in long_windows:
            # Skip invalid combinations
            if short_window >= long_window:
                continue
            
            logger.info(f"Testing parameters: short_window={short_window}, long_window={long_window}")
            
            # Initialize strategy and engine
            strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
            engine = BacktestEngine(initial_capital=initial_capital)
            
            # Run backtest
            metrics = engine.run(strategy, data)
            
            # Save results
            result = {
                'short_window': short_window,
                'long_window': long_window,
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor']
            }
            
            results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_path / f"optimization_{symbol}_{start_date}_{end_date}.csv", index=False)
    
    # Find best parameters by Sharpe ratio
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    # Find best parameters by total return
    best_return = results_df.loc[results_df['total_return'].idxmax()]
    
    # Create heatmap of Sharpe ratios
    plt.figure(figsize=(12, 10))
    pivot = results_df.pivot_table(index='short_window', columns='long_window', values='sharpe_ratio')
    plt.imshow(pivot, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Long Window')
    plt.ylabel('Short Window')
    plt.title(f'Sharpe Ratio Heatmap - {symbol}')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.savefig(output_path / f"heatmap_sharpe_{symbol}.png")
    
    # Create heatmap of returns
    plt.figure(figsize=(12, 10))
    pivot = results_df.pivot_table(index='short_window', columns='long_window', values='total_return')
    plt.imshow(pivot, cmap='viridis')
    plt.colorbar(label='Total Return')
    plt.xlabel('Long Window')
    plt.ylabel('Short Window')
    plt.title(f'Total Return Heatmap - {symbol}')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.savefig(output_path / f"heatmap_return_{symbol}.png")
    
    logger.info(f"Optimization completed. Results saved to {output_path}")
    logger.info(f"Best parameters by Sharpe ratio: short_window={best_sharpe['short_window']}, long_window={best_sharpe['long_window']}, sharpe={best_sharpe['sharpe_ratio']:.2f}")
    logger.info(f"Best parameters by total return: short_window={best_return['short_window']}, long_window={best_return['long_window']}, return={best_return['total_return']:.2%}")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest for moving average crossover strategy')
    
    # Basic arguments
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    
    # Strategy parameters
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
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    
    args = parser.parse_args()
    
    if args.optimize:
        # Run parameter optimization
        run_parameter_optimization(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            output_dir=args.report_dir + '/optimization'
        )
    else:
        # Run single backtest
        run_backtest(
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
            report_dir=args.report_dir
        )
