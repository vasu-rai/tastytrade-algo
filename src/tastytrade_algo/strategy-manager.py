# src/tastytrade_algo/strategy_manager.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Type
import json
from pathlib import Path

# Import strategy base class
from .strategy import Strategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manager for multiple trading strategies.
    
    This class handles the registration, allocation, and execution of
    multiple trading strategies across different symbols.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        data_dir: str = 'data',
        allocation_method: str = 'equal'  # 'equal', 'performance', 'custom'
    ):
        """
        Initialize the strategy manager.
        
        Args:
            config_file: Optional path to strategy configuration file
            data_dir: Directory for strategy data
            allocation_method: Method for allocating capital to strategies
        """
        self.strategies = {}  # name -> strategy object
        self.strategy_allocations = {}  # name -> allocation percentage
        self.strategy_performance = {}  # name -> performance metrics
        self.data_dir = Path(data_dir)
        self.allocation_method = allocation_method
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        
        logger.info("Initialized strategy manager")
    
    def register_strategy(
        self,
        strategy_name: str,
        strategy_class: Type[Strategy],
        strategy_params: Dict[str, Any],
        symbols: List[str],
        allocation: float = None
    ) -> None:
        """
        Register a strategy with the manager.
        
        Args:
            strategy_name: Unique name for the strategy
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for strategy initialization
            symbols: List of symbols this strategy will trade
            allocation: Optional allocation percentage (0-1.0)
        """
        if strategy_name in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' already exists, overwriting")
        
        # Initialize the strategy
        strategy = strategy_class(**strategy_params)
        
        # Store the strategy and its configuration
        self.strategies[strategy_name] = {
            'strategy': strategy,
            'class': strategy_class.__name__,
            'params': strategy_params,
            'symbols': symbols,
            'active': True  # Whether the strategy is active
        }
        
        # Set allocation if provided
        if allocation is not None:
            self.strategy_allocations[strategy_name] = allocation
        
        logger.info(f"Registered strategy '{strategy_name}' for symbols {symbols}")
        
        # Update allocations
        self._update_allocations()
    
    def deregister_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from the manager.
        
        Args:
            strategy_name: Name of the strategy to remove
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return False
        
        # Remove strategy
        del self.strategies[strategy_name]
        
        # Remove allocation if exists
        if strategy_name in self.strategy_allocations:
            del self.strategy_allocations[strategy_name]
        
        # Remove performance if exists
        if strategy_name in self.strategy_performance:
            del self.strategy_performance[strategy_name]
        
        # Update allocations
        self._update_allocations()
        
        logger.info(f"Deregistered strategy '{strategy_name}'")
        return True
    
    def activate_strategy(self, strategy_name: str) -> bool:
        """
        Activate a strategy.
        
        Args:
            strategy_name: Name of the strategy to activate
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return False
        
        self.strategies[strategy_name]['active'] = True
        
        # Update allocations
        self._update_allocations()
        
        logger.info(f"Activated strategy '{strategy_name}'")
        return True
    
    def deactivate_strategy(self, strategy_name: str) -> bool:
        """
        Deactivate a strategy.
        
        Args:
            strategy_name: Name of the strategy to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return False
        
        self.strategies[strategy_name]['active'] = False
        
        # Update allocations
        self._update_allocations()
        
        logger.info(f"Deactivated strategy '{strategy_name}'")
        return True
    
    def set_allocation(self, strategy_name: str, allocation: float) -> bool:
        """
        Set custom allocation for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            allocation: Allocation percentage (0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return False
        
        if allocation < 0 or allocation > 1.0:
            logger.warning(f"Allocation must be between 0 and 1.0")
            return False
        
        self.strategy_allocations[strategy_name] = allocation
        
        # Set allocation method to custom
        self.allocation_method = 'custom'
        
        # Update allocations
        self._update_allocations()
        
        logger.info(f"Set allocation for '{strategy_name}' to {allocation:.2%}")
        return True
    
    def set_allocation_method(self, method: str) -> bool:
        """
        Set the allocation method.
        
        Args:
            method: Allocation method ('equal', 'performance', 'custom')
            
        Returns:
            True if successful, False otherwise
        """
        valid_methods = ['equal', 'performance', 'custom']
        
        if method not in valid_methods:
            logger.warning(f"Invalid allocation method '{method}'. "
                           f"Valid methods: {valid_methods}")
            return False
        
        self.allocation_method = method
        
        # Update allocations
        self._update_allocations()
        
        logger.info(f"Set allocation method to '{method}'")
        return True
    
    def _update_allocations(self) -> None:
        """Update strategy allocations based on the current method."""
        # Get active strategies
        active_strategies = [name for name, info in self.strategies.items() 
                            if info['active']]
        
        if not active_strategies:
            logger.warning("No active strategies")
            return
        
        if self.allocation_method == 'equal':
            # Equal allocation across active strategies
            allocation = 1.0 / len(active_strategies)
            
            for name in active_strategies:
                self.strategy_allocations[name] = allocation
                
        elif self.allocation_method == 'performance':
            # Allocate based on performance
            if not self.strategy_performance:
                # No performance data, use equal allocation
                logger.warning("No performance data available, using equal allocation")
                allocation = 1.0 / len(active_strategies)
                
                for name in active_strategies:
                    self.strategy_allocations[name] = allocation
            else:
                # Use Sharpe ratio for allocation
                sharpes = {}
                
                for name in active_strategies:
                    if name in self.strategy_performance:
                        sharpe = self.strategy_performance[name].get('sharpe_ratio', 0.0)
                        # Ensure Sharpe is positive for allocation
                        sharpes[name] = max(sharpe, 0.01)
                    else:
                        # No performance data, use minimum allocation
                        sharpes[name] = 0.01
                
                # Normalize
                total = sum(sharpes.values())
                
                if total > 0:
                    for name, sharpe in sharpes.items():
                        self.strategy_allocations[name] = sharpe / total
                else:
                    # Fallback to equal allocation
                    allocation = 1.0 / len(active_strategies)
                    
                    for name in active_strategies:
                        self.strategy_allocations[name] = allocation
                        
        elif self.allocation_method == 'custom':
            # Custom allocations - just make sure all active strategies have an allocation
            for name in active_strategies:
                if name not in self.strategy_allocations:
                    logger.warning(f"Strategy '{name}' has no custom allocation, setting to 0")
                    self.strategy_allocations[name] = 0.0
            
            # Normalize custom allocations to sum to 1.0
            total = sum(self.strategy_allocations[name] for name in active_strategies)
            
            if total > 0:
                for name in active_strategies:
                    self.strategy_allocations[name] = self.strategy_allocations[name] / total
        
        # Ensure inactive strategies have 0 allocation
        for name in self.strategies:
            if name not in active_strategies:
                self.strategy_allocations[name] = 0.0
        
        logger.debug(f"Updated strategy allocations: {self.strategy_allocations}")
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return False
        
        self.strategy_performance[strategy_name] = performance_metrics
        
        # If using performance-based allocation, update allocations
        if self.allocation_method == 'performance':
            self._update_allocations()
        
        logger.info(f"Updated performance metrics for '{strategy_name}'")
        return True
    
    def get_strategy_signals(
        self,
        strategy_name: str,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            data: Dictionary of price data (symbol -> DataFrame)
            
        Returns:
            Dictionary of signal DataFrames (symbol -> DataFrame with signals)
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return {}
        
        strategy_info = self.strategies[strategy_name]
        
        if not strategy_info['active']:
            logger.warning(f"Strategy '{strategy_name}' is not active")
            return {}
        
        strategy = strategy_info['strategy']
        symbols = strategy_info['symbols']
        
        signals = {}
        
        for symbol in symbols:
            if symbol in data:
                try:
                    symbol_data = data[symbol].copy()
                    signals[symbol] = strategy.generate_signals(symbol_data)
                    logger.debug(f"Generated signals for '{strategy_name}' on {symbol}")
                except Exception as e:
                    logger.error(f"Error generating signals for '{strategy_name}' on {symbol}: {e}")
        
        return signals
    
    def get_all_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate signals for all active strategies.
        
        Args:
            data: Dictionary of price data (symbol -> DataFrame)
            
        Returns:
            Dictionary of signal DataFrames (strategy_name -> symbol -> DataFrame)
        """
        all_signals = {}
        
        for strategy_name, strategy_info in self.strategies.items():
            if strategy_info['active']:
                signals = self.get_strategy_signals(strategy_name, data)
                
                if signals:
                    all_signals[strategy_name] = signals
        
        return all_signals
    
    def get_combined_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Combine signals from all active strategies.
        
        This method weights the signals according to strategy allocations.
        
        Args:
            data: Dictionary of price data (symbol -> DataFrame)
            
        Returns:
            Dictionary of combined signal DataFrames (symbol -> DataFrame)
        """
        # Get signals from all strategies
        all_signals = self.get_all_signals(data)
        
        if not all_signals:
            logger.warning("No signals generated")
            return {}
        
        # Combine signals for each symbol
        combined_signals = {}
        
        # Get all unique symbols across all strategies
        all_symbols = set()
        for strategy_signals in all_signals.values():
            all_symbols.update(strategy_signals.keys())
        
        for symbol in all_symbols:
            # Create a DataFrame with dates
            dates = set()
            for strategy_name, strategy_signals in all_signals.items():
                if symbol in strategy_signals:
                    dates.update(strategy_signals[symbol].index)
            
            if not dates:
                continue
            
            dates = sorted(dates)
            combined_df = pd.DataFrame(index=dates)
            
            # Add signals from each strategy
            for strategy_name, strategy_signals in all_signals.items():
                if symbol in strategy_signals:
                    # Get allocation for this strategy
                    allocation = self.strategy_allocations.get(strategy_name, 0.0)
                    
                    if allocation > 0:
                        signal_df = strategy_signals[symbol]
                        
                        # Resample to match combined dates if needed
                        if not signal_df.index.equals(combined_df.index):
                            signal_df = signal_df.reindex(combined_df.index, method='ffill')
                        
                        # Weight the signals by allocation
                        weighted_signal = signal_df['signal'] * allocation
                        
                        # Add to combined DataFrame
                        col_name = f"{strategy_name}_signal"
                        combined_df[col_name] = weighted_signal
            
            # Calculate combined signal
            signal_cols = [col for col in combined_df.columns if col.endswith('_signal')]
            
            if signal_cols:
                combined_df['signal'] = combined_df[signal_cols].sum(axis=1)
                
                # Normalize to -1, 0, 1
                combined_df['signal'] = np.sign(combined_df['signal'])
                
                # Calculate position changes
                combined_df['position_change'] = combined_df['signal'].diff().fillna(0)
                
                combined_signals[symbol] = combined_df
        
        return combined_signals
    
    def save_config(self, filename: str) -> bool:
        """
        Save strategy configuration to a file.
        
        Args:
            filename: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        config = {
            'allocation_method': self.allocation_method,
            'strategies': {},
            'allocations': self.strategy_allocations,
            'performance': self.strategy_performance
        }
        
        # Save strategy configurations
        for name, info in self.strategies.items():
            config['strategies'][name] = {
                'class': info['class'],
                'params': info['params'],
                'symbols': info['symbols'],
                'active': info['active']
            }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Strategy configuration saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving strategy configuration: {e}")
            return False
    
    def load_config(self, filename: str) -> bool:
        """
        Load strategy configuration from a file.
        
        Args:
            filename: Path to the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load allocation method
            if 'allocation_method' in config:
                self.allocation_method = config['allocation_method']
            
            # Load allocations
            if 'allocations' in config:
                self.strategy_allocations = config['allocations']
            
            # Load performance
            if 'performance' in config:
                self.strategy_performance = config['performance']
            
            # Load strategies
            if 'strategies' in config:
                for name, info in config['strategies'].items():
                    try:
                        # Import the strategy class
                        class_name = info['class']
                        strategy_class = self._import_strategy_class(class_name)
                        
                        # Register the strategy
                        self.register_strategy(
                            strategy_name=name,
                            strategy_class=strategy_class,
                            strategy_params=info['params'],
                            symbols=info['symbols'],
                            allocation=None  # Use the loaded allocations
                        )
                        
                        # Set active state
                        if not info.get('active', True):
                            self.deactivate_strategy(name)
                        
                    except Exception as e:
                        logger.error(f"Error loading strategy '{name}': {e}")
            
            logger.info(f"Strategy configuration loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading strategy configuration: {e}")
            return False
    
    def _import_strategy_class(self, class_name: str) -> Type[Strategy]:
        """
        Import a strategy class by name.
        
        Args:
            class_name: Name of the strategy class
            
        Returns:
            Strategy class
        """
        # Check if the class is in our module
        from . import strategy as strategy_module
        
        if hasattr(strategy_module, class_name):
            return getattr(strategy_module, class_name)
        
        # Try to import dynamically
        try:
            import importlib
            module_path = f"tastytrade_algo.strategy.{class_name.lower()}"
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            logger.error(f"Error importing strategy class '{class_name}': {e}")
            raise ValueError(f"Strategy class '{class_name}' not found")
    
    def get_allocation_for_symbol(self, symbol: str) -> Dict[str, float]:
        """
        Get the allocation for a specific symbol across strategies.
        
        Args:
            symbol: Symbol to get allocations for
            
        Returns:
            Dictionary of strategy name -> allocation percentage
        """
        allocations = {}
        
        for name, info in self.strategies.items():
            if info['active'] and symbol in info['symbols']:
                allocations[name] = self.strategy_allocations.get(name, 0.0)
        
        return allocations
    
    def get_symbols_for_strategy(self, strategy_name: str) -> List[str]:
        """
        Get the symbols traded by a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            List of symbols
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return []
        
        return self.strategies[strategy_name]['symbols']
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all symbols traded by any strategy.
        
        Returns:
            List of unique symbols
        """
        symbols = set()
        
        for info in self.strategies.values():
            symbols.update(info['symbols'])
        
        return sorted(list(symbols))
    
    def get_active_strategies(self) -> List[str]:
        """
        Get names of all active strategies.
        
        Returns:
            List of strategy names
        """
        return [name for name, info in self.strategies.items() if info['active']]
    
    def get_strategy_info(self, strategy_name: str) -> Dict:
        """
        Get information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of strategy information
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' does not exist")
            return {}
        
        info = self.strategies[strategy_name].copy()
        
        # Remove the actual strategy object
        if 'strategy' in info:
            del info['strategy']
        
        # Add allocation and performance if available
        info['allocation'] = self.strategy_allocations.get(strategy_name, 0.0)
        
        if strategy_name in self.strategy_performance:
            info['performance'] = self.strategy_performance[strategy_name]
        
        return info