    def _generate_and_execute_signals(self) -> None:
        """Generate signals and execute trades based on current data."""
        if not self.strategy_manager or not self.current_data:
            return
        
        # Generate signals
        signals = self.strategy_manager.get_combined_signals(self.current_data)
        
        if not signals:
            logger.debug("No signals generated")
            return
        
        # Execute signals
        for symbol, signal_df in signals.items():
            if symbol in self.current_data and not signal_df.empty:
                # Get the latest signal
                latest_signal = signal_df.iloc[-1]
                
                # Check if we have a position change
                if 'position_change' in latest_signal and latest_signal['position_change'] != 0:
                    # Get current price
                    current_price = self.current_data[symbol]['close'].iloc[-1]
                    current_time = self.last_update or datetime.now()
                    
                    # Check the type of signal
                    if latest_signal['position_change'] > 0:  # Buy signal
                        # Open position if we don't have one
                        if self.portfolio_manager and symbol not in self.portfolio_manager.positions:
                            logger.info(f"BUY signal for {symbol} at {current_price}")
                            
                            # Calculate position size (automatic in portfolio manager)
                            self.portfolio_manager.open_position(
                                symbol=symbol,
                                quantity=None,  # Automatic calculation
                                price=current_price,
                                date=current_time,
                                strategy_name="combined",
                                metadata={"signal_strength": latest_signal['signal']}
                            )
                    
                    elif latest_signal['position_change'] < 0:  # Sell signal
                        # Close position if we have one
                        if self.portfolio_manager and symbol in self.portfolio_manager.positions:
                            logger.info(f"SELL signal for {symbol} at {current_price}")
                            
                            self.portfolio_manager.close_position(
                                symbol=symbol,
                                price=current_price,
                                date=current_time,
                                reason="signal"
                            )
    
    def _save_state(self) -> None:
        """Save the current state of the trading engine."""
        # Save strategy configuration
        if self.strategy_manager:
            self.strategy_manager.save_config(str(self.config_dir / 'strategies.json'))
        
        # Save portfolio state
        if self.portfolio_manager:
            self.portfolio_manager.save_state(str(self.output_dir / 'portfolio_state.json'))
        
        # Save engine configuration
        self.save_config(str(self.config_dir / 'engine.json'))
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading engine.
        
        Returns:
            Dictionary of status information
        """
        status = {
            'mode': self.mode,
            'is_running': self.is_running,
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else None,
            'api_connected': self.api is not None and hasattr(self.api, 'token') and self.api.token is not None,
            'portfolio': {},
            'strategies': {}
        }
        
        # Add portfolio information
        if self.portfolio_manager:
            status['portfolio'] = {
                'initial_capital': self.portfolio_manager.initial_capital,
                'current_value': self.portfolio_manager.get_portfolio_value(),
                'cash': self.portfolio_manager.cash,
                'open_positions': len(self.portfolio_manager.positions),
                'closed_positions': len(self.portfolio_manager.closed_positions)
            }
            
            # Add performance metrics
            metrics = self.portfolio_manager.get_performance_metrics()
            if metrics:
                status['portfolio']['metrics'] = metrics
        
        # Add strategy information
        if self.strategy_manager:
            status['strategies'] = {
                'allocation_method': self.strategy_manager.allocation_method,
                'active_strategies': self.strategy_manager.get_active_strategies(),
                'total_strategies': len(self.strategy_manager.strategies),
                'symbols': self.strategy_manager.get_all_symbols()
            }
        
        return status
    
    def get_portfolio_report(self, save_dir: Optional[str] = None) -> str:
        """
        Generate a report of the portfolio.
        
        Args:
            save_dir: Optional directory to save report files
            
        Returns:
            Portfolio report as a string
        """
        if not self.portfolio_manager:
            return "Portfolio manager not initialized"
        
        return self.portfolio_manager.generate_report(save_dir)
    
    def get_strategy_report(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about strategies.
        
        Args:
            strategy_name: Optional name of specific strategy to report on
            
        Returns:
            Dictionary of strategy information
        """
        if not self.strategy_manager:
            return {"error": "Strategy manager not initialized"}
        
        if strategy_name:
            return self.strategy_manager.get_strategy_info(strategy_name)
        else:
            return {name: self.strategy_manager.get_strategy_info(name) 
                    for name in self.strategy_manager.strategies}
    
    def execute_trade(
        self,
        symbol: str,
        action: str,  # 'buy' or 'sell'
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = 'market'  # 'market', 'limit', 'stop', 'stop_limit'
    ) -> Dict[str, Any]:
        """
        Execute a manual trade.
        
        Args:
            symbol: Symbol to trade
            action: Trade action ('buy' or 'sell')
            quantity: Number of shares/contracts (None for automatic sizing)
            price: Price for limit/stop orders (None for market price)
            order_type: Order type
            
        Returns:
            Dictionary with trade result
        """
        if not self.portfolio_manager:
            return {"error": "Portfolio manager not initialized"}
        
        current_time = datetime.now()
        
        # Get current price if not provided
        if price is None:
            if symbol in self.current_data and not self.current_data[symbol].empty:
                price = self.current_data[symbol]['close'].iloc[-1]
            else:
                return {"error": f"No price data available for {symbol}"}
        
        # Execute trade
        result = {"symbol": symbol, "action": action, "time": current_time, "success": False}
        
        try:
            if action.lower() == 'buy':
                position = self.portfolio_manager.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    date=current_time,
                    strategy_name="manual",
                    metadata={"order_type": order_type}
                )
                
                if position:
                    result["success"] = True
                    result["position"] = position.to_dict()
                else:
                    result["error"] = "Failed to open position"
            
            elif action.lower() == 'sell':
                position = self.portfolio_manager.close_position(
                    symbol=symbol,
                    price=price,
                    date=current_time,
                    reason="manual"
                )
                
                if position:
                    result["success"] = True
                    result["position"] = position.to_dict()
                else:
                    result["error"] = "Failed to close position"
            
            else:
                result["error"] = f"Invalid action: {action}"
        
        except Exception as e:
            result["error"] = str(e)
        
        # Log and return result
        if result["success"]:
            logger.info(f"Manual trade executed: {action} {symbol} at {price}")
        else:
            logger.error(f"Manual trade failed: {action} {symbol} - {result.get('error', 'Unknown error')}")
        
        return result
    
    def optimize_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        parameter_ranges: Dict[str, List[Any]],
        optimization_metric: str = 'sharpe_ratio',
        report_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            symbols: List of symbols to use for optimization
            start_date: Start date for optimization period
            end_date: End date for optimization period
            parameter_ranges: Dictionary of parameter names to lists of values
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            report_dir: Optional directory to save optimization reports
            
        Returns:
            Dictionary of optimization results
        """
        if not self.strategy_manager or strategy_name not in self.strategy_manager.strategies:
            return {"error": f"Strategy '{strategy_name}' not found"}
        
        if not self.backtest_engine:
            self.backtest_engine = BacktestEngine(
                initial_capital=self.portfolio_manager.initial_capital if self.portfolio_manager else 100000.0
            )
        
        # Create report directory
        if report_dir:
            report_path = Path(report_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"optimization_{strategy_name}_{timestamp}"
        
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Get historical data
        data = {}
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}")
            symbol_data = self.data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(symbol_data) > 0:
                data[symbol] = self.data_fetcher.prepare_data(symbol_data)
            else:
                logger.warning(f"No data available for {symbol}")
        
        if not data:
            logger.error("No data available for optimization")
            return {"error": "No data available"}
        
        # Get strategy class and base parameters
        strategy_info = self.strategy_manager.strategies[strategy_name]
        strategy_class = strategy_info['strategy'].__class__
        base_params = strategy_info['params']
        
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(itertools.product(*[parameter_ranges[name] for name in param_names]))
        
        # Store optimization results
        results = []
        
        # Run backtest for each parameter combination
        for values in param_values:
            params = base_params.copy()
            
            # Update parameters
            for name, value in zip(param_names, values):
                params[name] = value
            
            # Create strategy instance
            strategy = strategy_class(**params)
            
            # Run backtest for each symbol
            symbol_results = {}
            
            for symbol, symbol_data in data.items():
                try:
                    # Generate signals
                    signals = strategy.generate_signals(symbol_data)
                    
                    # Run backtest
                    self.backtest_engine.run(strategy, signals)
                    
                    # Store metrics
                    symbol_results[symbol] = self.backtest_engine.metrics.copy()
                except Exception as e:
                    logger.error(f"Error in backtest for {symbol} with params {params}: {e}")
            
            # Calculate average metrics across symbols
            if symbol_results:
                avg_metrics = {}
                
                for metric in next(iter(symbol_results.values())).keys():
                    avg_metrics[metric] = sum(result.get(metric, 0) for result in symbol_results.values()) / len(symbol_results)
                
                # Store result
                result = {
                    'params': params,
                    'metrics': avg_metrics,
                    'symbol_results': symbol_results
                }
                
                results.append(result)
                
                logger.info(f"Tested params {params}: {optimization_metric}={avg_metrics.get(optimization_metric, 0)}")
        
        # Find best result
        if results:
            best_result = max(results, key=lambda r: r['metrics'].get(optimization_metric, float('-inf')))
            
            # Save results
            with open(report_path / 'all_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            with open(report_path / 'best_result.json', 'w') as f:
                json.dump(best_result, f, indent=2)
            
            logger.info(f"Optimization completed. Best {optimization_metric}={best_result['metrics'].get(optimization_metric, 0)}")
            
            return {
                'best_result': best_result,
                'all_results': results,
                'report_dir': str(report_path)
            }
        else:
            logger.error("No optimization results")
            return {"error": "No optimization results"}
    
    def create_strategy(
        self,
        strategy_name: str,
        strategy_type: str,
        parameters: Dict[str, Any],
        symbols: List[str],
        allocation: float = None
    ) -> bool:
        """
        Create a new strategy.
        
        Args:
            strategy_name: Name for the new strategy
            strategy_type: Type of strategy (e.g., 'MovingAverageCrossover')
            parameters: Dictionary of strategy parameters
            symbols: List of symbols for this strategy
            allocation: Optional allocation percentage
            
        Returns:
            True if successful, False otherwise
        """
        if not self.strategy_manager:
            logger.error("Strategy manager not initialized")
            return False
        
        try:
            # Import strategy class
            strategy_class = self.strategy_manager._import_strategy_class(strategy_type)
            
            # Register strategy
            self.strategy_manager.register_strategy(
                strategy_name=strategy_name,
                strategy_class=strategy_class,
                strategy_params=parameters,
                symbols=symbols,
                allocation=allocation
            )
            
            # Save strategy configuration
            self.strategy_manager.save_config(str(self.config_dir / 'strategies.json'))
            
            logger.info(f"Created strategy '{strategy_name}' of type {strategy_type}")
            return True
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            return False
                        # src/tastytrade_algo/trading_engine.py
import pandas as pd
import numpy as np
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import traceback

# Import components
from .tastytrade_api import TastytradeAPI
from .strategy_manager import StrategyManager
from .portfolio import PortfolioManager
from .data_fetcher import DataFetcher
from .backtest import BacktestEngine

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine class.
    
    This class coordinates the entire trading system, including:
    - Strategy management
    - Portfolio management
    - Data fetching
    - Signal generation
    - Trade execution
    - Performance tracking
    
    It ties together all the components of the trading system.
    """
    
    def __init__(
        self,
        mode: str = 'backtest',  # 'backtest', 'paper', 'live'
        config_dir: str = 'config',
        data_dir: str = 'data',
        output_dir: str = 'output',
        config_file: Optional[str] = None
    ):
        """
        Initialize the trading engine.
        
        Args:
            mode: Trading mode ('backtest', 'paper', 'live')
            config_dir: Directory for configuration files
            data_dir: Directory for data files
            output_dir: Directory for output files
            config_file: Optional path to engine configuration file
        """
        self.mode = mode
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.api = None
        self.data_fetcher = None
        self.strategy_manager = None
        self.portfolio_manager = None
        self.backtest_engine = None
        
        # Runtime state
        self.is_running = False
        self.last_update = None
        self.current_data = {}  # symbol -> DataFrame
        self.current_positions = {}  # symbol -> position details
        
        # Load configuration if provided
        if config_file:
            self.load_config(config_file)
        else:
            self._init_default_components()
        
        logger.info(f"Initialized trading engine in {mode} mode")
    
    def _init_default_components(self):
        """Initialize default components based on the mode."""
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(
            cache_dir=str(self.data_dir / 'cache'),
            tastytrade_api=self.api
        )
        
        # Initialize strategy manager
        strategy_config = self.config_dir / 'strategies.json'
        if strategy_config.exists():
            self.strategy_manager = StrategyManager(
                config_file=str(strategy_config),
                data_dir=str(self.data_dir)
            )
        else:
            self.strategy_manager = StrategyManager(
                data_dir=str(self.data_dir)
            )
        
        # Initialize portfolio manager
        portfolio_state = self.output_dir / 'portfolio_state.json'
        self.portfolio_manager = PortfolioManager(
            initial_capital=100000.0,
            state_file=str(portfolio_state) if portfolio_state.exists() else None
        )
        
        # Initialize backtest engine if in backtest mode
        if self.mode == 'backtest':
            self.backtest_engine = BacktestEngine(
                initial_capital=self.portfolio_manager.initial_capital
            )
    
    def connect_api(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        environment: str = 'prod'  # 'prod' or 'cert'
    ) -> bool:
        """
        Connect to the Tastytrade API.
        
        Args:
            username: Tastytrade username (can be loaded from env)
            password: Tastytrade password (can be loaded from env)
            environment: API environment ('prod' or 'cert')
            
        Returns:
            True if connection successful, False otherwise
        """
        if not username:
            username = os.getenv('TASTYTRADE_USERNAME')
        
        if not password:
            password = os.getenv('TASTYTRADE_PASSWORD')
        
        if not username or not password:
            logger.error("API credentials not provided")
            return False
        
        try:
            logger.info(f"Connecting to Tastytrade API ({environment})")
            self.api = TastytradeAPI(username, password)
            
            # Update data fetcher with API
            if self.data_fetcher:
                self.data_fetcher.tastytrade_api = self.api
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            return False
    
    def load_config(self, config_file: str) -> bool:
        """
        Load engine configuration from a file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Set mode
            if 'mode' in config:
                self.mode = config['mode']
            
            # Connect to API if credentials are provided
            if 'api' in config:
                api_config = config['api']
                self.connect_api(
                    username=api_config.get('username'),
                    password=api_config.get('password'),
                    environment=api_config.get('environment', 'prod')
                )
            
            # Initialize data fetcher
            cache_dir = config.get('data_dir', str(self.data_dir / 'cache'))
            self.data_fetcher = DataFetcher(
                cache_dir=cache_dir,
                api_key=config.get('api_key'),
                tastytrade_api=self.api
            )
            
            # Initialize strategy manager
            strategy_config = config.get('strategy_config')
            if strategy_config:
                self.strategy_manager = StrategyManager(
                    config_file=strategy_config,
                    data_dir=str(self.data_dir),
                    allocation_method=config.get('allocation_method', 'equal')
                )
            else:
                self.strategy_manager = StrategyManager(
                    data_dir=str(self.data_dir)
                )
            
            # Initialize portfolio manager
            portfolio_config = config.get('portfolio', {})
            self.portfolio_manager = PortfolioManager(
                initial_capital=portfolio_config.get('initial_capital', 100000.0),
                max_capital_per_position=portfolio_config.get('max_capital_per_position', 0.1),
                max_positions=portfolio_config.get('max_positions', 10),
                portfolio_stop_loss=portfolio_config.get('portfolio_stop_loss', 0.2),
                state_file=portfolio_config.get('state_file')
            )
            
            # Initialize backtest engine if in backtest mode
            if self.mode == 'backtest':
                backtest_config = config.get('backtest', {})
                self.backtest_engine = BacktestEngine(
                    initial_capital=self.portfolio_manager.initial_capital,
                    commission=backtest_config.get('commission', 0.0),
                    slippage=backtest_config.get('slippage', 0.001)
                )
            
            logger.info(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            traceback.print_exc()
            return False
    
    def save_config(self, config_file: str) -> bool:
        """
        Save engine configuration to a file.
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                'mode': self.mode,
                'data_dir': str(self.data_dir),
                'output_dir': str(self.output_dir),
                'strategy_config': str(self.config_dir / 'strategies.json'),
                'allocation_method': self.strategy_manager.allocation_method if self.strategy_manager else 'equal',
                'portfolio': {
                    'initial_capital': self.portfolio_manager.initial_capital if self.portfolio_manager else 100000.0,
                    'max_capital_per_position': self.portfolio_manager.max_capital_per_position if self.portfolio_manager else 0.1,
                    'max_positions': self.portfolio_manager.max_positions if self.portfolio_manager else 10,
                    'portfolio_stop_loss': self.portfolio_manager.portfolio_stop_loss if self.portfolio_manager else 0.2,
                    'state_file': str(self.output_dir / 'portfolio_state.json')
                },
                'backtest': {
                    'commission': self.backtest_engine.commission if self.backtest_engine else 0.0,
                    'slippage': self.backtest_engine.slippage if self.backtest_engine else 0.001
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Also save strategy configuration
            if self.strategy_manager:
                self.strategy_manager.save_config(str(self.config_dir / 'strategies.json'))
            
            logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def run_backtest(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        symbols: Optional[List[str]] = None,
        report_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for the specified period and symbols.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            symbols: List of symbols to backtest (defaults to all)
            report_dir: Directory to save backtest reports
            
        Returns:
            Dictionary of backtest results
        """
        if self.mode != 'backtest':
            logger.warning(f"Switching mode from {self.mode} to backtest")
            self.mode = 'backtest'
            
            # Initialize backtest engine if needed
            if not self.backtest_engine:
                self.backtest_engine = BacktestEngine(
                    initial_capital=self.portfolio_manager.initial_capital
                )
        
        # Get symbols from strategy manager if not provided
        if not symbols and self.strategy_manager:
            symbols = self.strategy_manager.get_all_symbols()
        
        if not symbols:
            logger.error("No symbols to backtest")
            return {}
        
        logger.info(f"Starting backtest from {start_date} to {end_date} for {len(symbols)} symbols")
        
        # Create report directory
        if report_dir:
            report_path = Path(report_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"backtest_{timestamp}"
        
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Get historical data
        data = {}
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}")
            symbol_data = self.data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(symbol_data) > 0:
                data[symbol] = self.data_fetcher.prepare_data(symbol_data)
            else:
                logger.warning(f"No data available for {symbol}")
        
        if not data:
            logger.error("No data available for backtest")
            return {}
        
        # Update correlation matrix
        if self.portfolio_manager:
            self.portfolio_manager.update_correlation_matrix(data)
        
        # Generate combined signals
        signals = None
        if self.strategy_manager:
            logger.info("Generating signals from strategies")
            signals = self.strategy_manager.get_combined_signals(data)
        
        # Run backtest
        results = {}
        
        if signals:
            logger.info("Running backtest with strategy signals")
            
            # Run backtest for each symbol
            for symbol, signal_df in signals.items():
                if symbol in data:
                    # Combine data and signals
                    backtest_data = data[symbol].copy()
                    for col in signal_df.columns:
                        if col not in backtest_data.columns:
                            backtest_data[col] = signal_df[col]
                    
                    # Create a custom strategy for the backtest engine
                    from .strategy import Strategy
                    
                    class CombinedSignalStrategy(Strategy):
                        def generate_signals(self, data):
                            # Signals are already in the data
                            return data
                    
                    strategy = CombinedSignalStrategy()
                    
                    # Run backtest
                    logger.info(f"Running backtest for {symbol}")
                    self.backtest_engine.run(strategy, backtest_data)
                    
                    # Generate report
                    symbol_report_dir = report_path / symbol
                    symbol_report_dir.mkdir(parents=True, exist_ok=True)
                    
                    self.backtest_engine.generate_report(str(symbol_report_dir))
                    
                    # Store results
                    results[symbol] = self.backtest_engine.metrics
        else:
            logger.warning("No signals available for backtest")
        
        # Save backtest configuration
        backtest_config = {
            'start_date': start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d'),
            'end_date': end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d'),
            'symbols': symbols,
            'initial_capital': self.backtest_engine.initial_capital,
            'commission': self.backtest_engine.commission,
            'slippage': self.backtest_engine.slippage
        }
        
        with open(report_path / 'config.json', 'w') as f:
            json.dump(backtest_config, f, indent=2)
        
        # Generate combined report
        if results:
            combined_results = {
                'symbols': list(results.keys()),
                'metrics': {
                    'total_return': sum(r.get('total_return', 0) for r in results.values()) / len(results),
                    'annual_return': sum(r.get('annual_return', 0) for r in results.values()) / len(results),
                    'sharpe_ratio': sum(r.get('sharpe_ratio', 0) for r in results.values()) / len(results),
                    'max_drawdown': max(r.get('max_drawdown', 0) for r in results.values()),
                    'win_rate': sum(r.get('win_rate', 0) for r in results.values()) / len(results),
                    'profit_factor': sum(r.get('profit_factor', 0) for r in results.values()) / len(results)
                }
            }
            
            with open(report_path / 'combined_results.json', 'w') as f:
                json.dump(combined_results, f, indent=2)
            
            results['combined'] = combined_results
        
        logger.info(f"Backtest completed, results saved to {report_path}")
        return results
    
    def run_paper_trading(
        self,
        duration: Optional[int] = None,  # Duration in seconds, None for continuous
        update_interval: int = 60  # Update interval in seconds
    ) -> None:
        """
        Run paper trading for the specified duration.
        
        Args:
            duration: Duration in seconds (None for continuous)
            update_interval: Update interval in seconds
        """
        if self.mode != 'paper':
            logger.warning(f"Switching mode from {self.mode} to paper")
            self.mode = 'paper'
        
        if not self.strategy_manager:
            logger.error("Strategy manager not initialized")
            return
        
        if not self.portfolio_manager:
            logger.error("Portfolio manager not initialized")
            return
        
        # Get symbols from strategy manager
        symbols = self.strategy_manager.get_all_symbols()
        
        if not symbols:
            logger.error("No symbols to trade")
            return
        
        logger.info(f"Starting paper trading for {len(symbols)} symbols")
        
        # Set running flag
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Check if duration is exceeded
                if duration and (current_time - start_time) > duration:
                    logger.info(f"Trading duration ({duration}s) exceeded")
                    break
                
                # Update data and generate signals
                try:
                    self._update_data(symbols)
                    self._generate_and_execute_signals()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    traceback.print_exc()
                
                # Save state
                self._save_state()
                
                # Wait for next update
                sleep_time = max(0, update_interval - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            # Reset running flag
            self.is_running = False
            
            # Save final state
            self._save_state()
            
            logger.info("Paper trading stopped")
    
    def run_live_trading(
        self,
        duration: Optional[int] = None,  # Duration in seconds, None for continuous
        update_interval: int = 60  # Update interval in seconds
    ) -> None:
        """
        Run live trading for the specified duration.
        
        Args:
            duration: Duration in seconds (None for continuous)
            update_interval: Update interval in seconds
        """
        if self.mode != 'live':
            logger.warning(f"Switching mode from {self.mode} to live")
            self.mode = 'live'
        
        if not self.api:
            logger.error("API not connected")
            return
        
        if not self.strategy_manager:
            logger.error("Strategy manager not initialized")
            return
        
        if not self.portfolio_manager:
            logger.error("Portfolio manager not initialized")
            return
        
        # Get symbols from strategy manager
        symbols = self.strategy_manager.get_all_symbols()
        
        if not symbols:
            logger.error("No symbols to trade")
            return
        
        logger.info(f"Starting live trading for {len(symbols)} symbols")
        
        # Set running flag
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Check if duration is exceeded
                if duration and (current_time - start_time) > duration:
                    logger.info(f"Trading duration ({duration}s) exceeded")
                    break
                
                # Update data and generate signals
                try:
                    self._update_data(symbols)
                    self._generate_and_execute_signals()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    traceback.print_exc()
                
                # Save state
                self._save_state()
                
                # Wait for next update
                sleep_time = max(0, update_interval - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            # Reset running flag
            self.is_running = False
            
            # Save final state
            self._save_state()
            
            logger.info("Live trading stopped")
    
    def stop_trading(self) -> None:
        """Stop any running trading process."""
        self.is_running = False
        logger.info("Trading stop requested")
    
    def _update_data(self, symbols: List[str]) -> None:
        """
        Update current data for the specified symbols.
        
        Args:
            symbols: List of symbols to update
        """
        current_time = datetime.now()
        self.last_update = current_time
        
        for symbol in symbols:
            try:
                # For paper/live trading, get recent data
                if self.mode in ['paper', 'live']:
                    # Calculate start date (1 year back for sufficient history)
                    start_date = (current_time - timedelta(days=365)).strftime('%Y-%m-%d')
                    end_date = current_time.strftime('%Y-%m-%d')
                    
                    # Fetch data
                    data = self.data_fetcher.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        force_download=(self.mode == 'live')  # Force download for live trading
                    )
                    
                    if len(data) > 0:
                        self.current_data[symbol] = self.data_fetcher.prepare_data(data)
                        logger.debug(f"Updated data for {symbol}: {len(self.current_data[symbol])} records")
                    else:
                        logger.warning(f"No data available for {symbol}")
            
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")
        
        # Update portfolio with current prices
        if self.portfolio_manager and self.current_data:
            # Get the latest prices
            current_prices = {}
            for symbol, data in self.current_data.items():
                if not data.empty:
                    current_prices[symbol] = data['close'].iloc[-1]
            
            # Update portfolio
            if current_prices:
                self.portfolio_manager.update_portfolio(current_prices, current_time)