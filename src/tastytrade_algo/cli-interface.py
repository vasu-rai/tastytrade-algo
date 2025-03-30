# src/tastytrade_algo/cli.py
import argparse
import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import components
from .trading_engine import TradingEngine
from .strategy import MovingAverageCrossover

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tastytrade Algorithmic Trading System')
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), 
                                help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to backtest')
    backtest_parser.add_argument('--report-dir', type=str, help='Directory to save backtest reports')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--duration', type=int, help='Trading duration in seconds')
    paper_parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--duration', type=int, help='Trading duration in seconds')
    live_parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    
    # Strategy creation command
    strategy_parser = subparsers.add_parser('create-strategy', help='Create a new strategy')
    strategy_parser.add_argument('--name', type=str, required=True, help='Strategy name')
    strategy_parser.add_argument('--type', type=str, required=True, help='Strategy type')
    strategy_parser.add_argument('--symbols', type=str, nargs='+', required=True, help='Symbols for the strategy')
    strategy_parser.add_argument('--params', type=str, required=True, help='Strategy parameters (JSON)')
    strategy_parser.add_argument('--allocation', type=float, help='Strategy allocation (0-1)')
    
    # Strategy optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    optimize_parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    optimize_parser.add_argument('--symbols', type=str, nargs='+', required=True, help='Symbols for optimization')
    optimize_parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), 
                               help='End date (YYYY-MM-DD)')
    optimize_parser.add_argument('--param-ranges', type=str, required=True, help='Parameter ranges (JSON)')
    optimize_parser.add_argument('--metric', type=str, default='sharpe_ratio', help='Optimization metric')
    optimize_parser.add_argument('--report-dir', type=str, help='Directory to save optimization reports')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    
    # Report commands
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--type', type=str, required=True, choices=['portfolio', 'strategy'], 
                             help='Report type')
    report_parser.add_argument('--strategy', type=str, help='Strategy name (for strategy report)')
    report_parser.add_argument('--output-dir', type=str, help='Directory to save reports')
    
    return parser.parse_args()

def run_backtest(engine: TradingEngine, args):
    """Run backtest command."""
    symbols = args.symbols
    
    # If symbols not provided, use all available symbols
    if not symbols and engine.strategy_manager:
        symbols = engine.strategy_manager.get_all_symbols()
    
    if not symbols:
        print("No symbols specified and no strategies configured")
        return
    
    print(f"Running backtest from {args.start_date} to {args.end_date} for symbols: {symbols}")
    
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
        report_dir=args.report_dir
    )
    
    if results:
        if 'combined' in results:
            print("\nCombined backtest results:")
            for metric, value in results['combined']['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nBacktest completed. Full results saved to {args.report_dir}")
    else:
        print("Backtest failed. Check logs for details.")

def run_paper_trading(engine: TradingEngine, args):
    """Run paper trading command."""
    print(f"Starting paper trading with {args.interval}s update interval")
    if args.duration:
        print(f"Trading will run for {args.duration}s")
    
    try:
        engine.run_paper_trading(
            duration=args.duration,
            update_interval=args.interval
        )
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
        engine.stop_trading()

def run_live_trading(engine: TradingEngine, args):
    """Run live trading command."""
    if not engine.api:
        print("API not connected. Please check your configuration.")
        return
    
    print(f"Starting LIVE trading with {args.interval}s update interval")
    if args.duration:
        print(f"Trading will run for {args.duration}s")
    
    confirm = input("Are you sure you want to start LIVE trading? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Live trading cancelled")
        return
    
    try:
        engine.run_live_trading(
            duration=args.duration,
            update_interval=args.interval
        )
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
        engine.stop_trading()

def create_strategy(engine: TradingEngine, args):
    """Create a new strategy."""
    try:
        # Parse parameters JSON
        params = json.loads(args.params)
        
        success = engine.create_strategy(
            strategy_name=args.name,
            strategy_type=args.type,
            parameters=params,
            symbols=args.symbols,
            allocation=args.allocation
        )
        
        if success:
            print(f"Strategy '{args.name}' created successfully")
        else:
            print(f"Failed to create strategy. Check logs for details.")
    except json.JSONDecodeError:
        print(f"Error parsing parameters JSON: {args.params}")
    except Exception as e:
        print(f"Error creating strategy: {e}")

def optimize_strategy(engine: TradingEngine, args):
    """Optimize strategy parameters."""
    try:
        # Parse parameter ranges JSON
        param_ranges = json.loads(args.param_ranges)
        
        results = engine.optimize_strategy(
            strategy_name=args.strategy,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            parameter_ranges=param_ranges,
            optimization_metric=args.metric,
            report_dir=args.report_dir
        )
        
        if 'error' in results:
            print(f"Optimization failed: {results['error']}")
        else:
            best_result = results['best_result']
            
            print("\nOptimization completed successfully!")
            print("\nBest parameters:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")
            
            print("\nPerformance metrics:")
            for metric, value in best_result['metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nFull results saved to {results['report_dir']}")
    except json.JSONDecodeError:
        print(f"Error parsing parameter ranges JSON: {args.param_ranges}")
    except Exception as e:
        print(f"Error during optimization: {e}")

def show_status(engine: TradingEngine, args):
    """Show system status."""
    status = engine.get_status()
    
    print("\n=== SYSTEM STATUS ===")
    print(f"Mode: {status['mode']}")
    print(f"Running: {status['is_running']}")
    print(f"Last update: {status['last_update']}")
    print(f"API connected: {status['api_connected']}")
    
    if 'portfolio' in status:
        print("\n=== PORTFOLIO ===")
        print(f"Initial capital: ${status['portfolio']['initial_capital']:.2f}")
        print(f"Current value: ${status['portfolio']['current_value']:.2f}")
        print(f"Cash: ${status['portfolio']['cash']:.2f}")
        print(f"Open positions: {status['portfolio']['open_positions']}")
        print(f"Closed positions: {status['portfolio']['closed_positions']}")
        
        if 'metrics' in status['portfolio']:
            print("\n=== PERFORMANCE ===")
            for metric, value in status['portfolio']['metrics'].items():
                print(f"{metric}: {value:.4f}")
    
    if 'strategies' in status:
        print("\n=== STRATEGIES ===")
        print(f"Allocation method: {status['strategies']['allocation_method']}")
        print(f"Active strategies: {status['strategies']['active_strategies']}")
        print(f"Total strategies: {status['strategies']['total_strategies']}")
        print(f"Symbols: {status['strategies']['symbols']}")

def generate_report(engine: TradingEngine, args):
    """Generate reports."""
    if args.type == 'portfolio':
        report = engine.get_portfolio_report(args.output_dir)
        print(report)
        if args.output_dir:
            print(f"Portfolio report saved to {args.output_dir}")
    elif args.type == 'strategy':
        if args.strategy:
            strategy_info = engine.get_strategy_report(args.strategy)
            
            print(f"\n=== STRATEGY: {args.strategy} ===")
            for key, value in strategy_info.items():
                if key != 'performance':
                    print(f"{key}: {value}")
            
            if 'performance' in strategy_info:
                print("\n=== PERFORMANCE ===")
                for metric, value in strategy_info['performance'].items():
                    print(f"{metric}: {value:.4f}")
        else:
            strategies = engine.get_strategy_report()
            
            print("\n=== STRATEGIES ===")
            for name, info in strategies.items():
                print(f"\n{name}:")
                for key, value in info.items():
                    if key != 'performance':
                        print(f"  {key}: {value}")
                
                if 'performance' in info:
                    print(f"  Performance:")
                    for metric, value in info['performance'].items():
                        print(f"    {metric}: {value:.4f}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create trading engine
    engine = TradingEngine(config_file=args.config)
    
    # Execute command
    if args.command == 'backtest':
        run_backtest(engine, args)
    elif args.command == 'paper':
        run_paper_trading(engine, args)
    elif args.command == 'live':
        run_live_trading(engine, args)
    elif args.command == 'create-strategy':
        create_strategy(engine, args)
    elif args.command == 'optimize':
        optimize_strategy(engine, args)
    elif args.command == 'status':
        show_status(engine, args)
    elif args.command == 'report':
        generate_report(engine, args)
    else:
        print("Please specify a command. Run with --help for usage information.")

if __name__ == '__main__':
    main()
