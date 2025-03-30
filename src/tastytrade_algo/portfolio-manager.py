# src/tastytrade_algo/portfolio.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class Position:
    """Class representing a position in a security."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_date: Union[str, datetime],
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_id: Optional[str] = None
    ):
        """
        Initialize a position.
        
        Args:
            symbol: Symbol of the security
            quantity: Number of shares/contracts (negative for short)
            entry_price: Entry price per share/contract
            entry_date: Date of entry
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            position_id: Optional unique ID for the position
        """
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        
        if isinstance(entry_date, str):
            self.entry_date = pd.to_datetime(entry_date)
        else:
            self.entry_date = entry_date
            
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_id = position_id or f"{symbol}_{entry_date.strftime('%Y%m%d%H%M%S')}"
        
        # Exit details (to be filled when position is closed)
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None
        self.pnl = None
        self.is_open = True
    
    def mark_to_market(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        return (current_price - self.entry_price) * self.quantity
    
    def close(
        self,
        exit_price: float,
        exit_date: Union[str, datetime],
        reason: str = "manual"
    ):
        """
        Close the position.
        
        Args:
            exit_price: Exit price per share/contract
            exit_date: Date of exit
            reason: Reason for exit (e.g., "stop_loss", "take_profit", "signal", "manual")
        """
        self.exit_price = exit_price
        
        if isinstance(exit_date, str):
            self.exit_date = pd.to_datetime(exit_date)
        else:
            self.exit_date = exit_date
            
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.is_open = False
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if the position should be closed based on stop loss or take profit.
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (should_close, reason)
        """
        if not self.is_open:
            return False, "already_closed"
        
        if self.stop_loss is not None:
            if (self.quantity > 0 and current_price <= self.stop_loss) or \
               (self.quantity < 0 and current_price >= self.stop_loss):
                return True, "stop_loss"
        
        if self.take_profit is not None:
            if (self.quantity > 0 and current_price >= self.take_profit) or \
               (self.quantity < 0 and current_price <= self.take_profit):
                return True, "take_profit"
        
        return False, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.strftime('%Y-%m-%d %H:%M:%S'),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_id': self.position_id,
            'exit_price': self.exit_price,
            'exit_date': self.exit_date.strftime('%Y-%m-%d %H:%M:%S') if self.exit_date else None,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'is_open': self.is_open
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        position = cls(
            symbol=data['symbol'],
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            entry_date=data['entry_date'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            position_id=data.get('position_id')
        )
        
        # Set exit details if available
        if 'exit_price' in data and data['exit_price'] is not None:
            position.exit_price = data['exit_price']
            position.exit_date = pd.to_datetime(data['exit_date'])
            position.exit_reason = data['exit_reason']
            position.pnl = data['pnl']
            position.is_open = data['is_open']
        
        return position

class PortfolioManager:
    """
    Portfolio manager for handling multiple positions and strategies.
    
    This class manages the allocation of capital across different strategies
    and symbols, keeps track of positions, and provides portfolio analytics.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_capital_per_position: float = 0.1,
        max_positions: int = 10,
        max_correlated_positions: int = 3,
        correlation_threshold: float = 0.7,
        portfolio_stop_loss: Optional[float] = 0.2,  # 20% drawdown triggers portfolio stop
        portfolio_target: Optional[float] = None,  # Target return
        risk_free_rate: float = 0.0,
        state_file: Optional[str] = None
    ):
        """
        Initialize the portfolio manager.
        
        Args:
            initial_capital: Starting capital
            max_capital_per_position: Maximum fraction of capital per position
            max_positions: Maximum number of open positions
            max_correlated_positions: Maximum number of correlated positions
            correlation_threshold: Threshold for considering positions correlated
            portfolio_stop_loss: Optional portfolio-wide stop loss (drawdown)
            portfolio_target: Optional portfolio-wide target return
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            state_file: Optional file to save/load portfolio state
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_capital_per_position = max_capital_per_position
        self.max_positions = max_positions
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        self.portfolio_stop_loss = portfolio_stop_loss
        self.portfolio_target = portfolio_target
        self.risk_free_rate = risk_free_rate
        self.state_file = state_file
        
        # Portfolio state
        self.positions = {}  # Current positions (symbol -> Position)
        self.closed_positions = []  # Historical closed positions
        self.cash = initial_capital  # Available cash
        
        # Performance tracking
        self.equity_curve = pd.Series([initial_capital], index=[pd.Timestamp.now()])
        self.returns = pd.Series([0.0], index=[pd.Timestamp.now()])
        
        # Correlation matrix for symbols
        self.correlation_matrix = pd.DataFrame()
        
        logger.info(f"Initialized portfolio manager with {initial_capital:.2f} capital")
        
        # Load state if file exists
        if state_file and Path(state_file).exists():
            self.load_state(state_file)
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: Union[str, datetime],
        strategy_name: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares/contracts (negative for short)
            price: Entry price
            date: Entry date
            strategy_name: Name of the strategy opening the position
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            metadata: Optional metadata about the position (e.g., strategy params)
            
        Returns:
            Position object if successful, None otherwise
        """
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}, not opening another one")
            return None
        
        # Check if we have too many positions
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached, not opening new position")
            return None
        
        # Check if we have too many correlated positions
        if len(self.correlation_matrix) > 0 and symbol in self.correlation_matrix.index:
            correlated_symbols = self._get_correlated_symbols(symbol)
            current_correlated = sum(1 for s in correlated_symbols if s in self.positions)
            
            if current_correlated >= self.max_correlated_positions:
                logger.warning(f"Maximum correlated positions ({self.max_correlated_positions}) reached")
                return None
        
        # Calculate position size if quantity is None
        if quantity is None:
            quantity = self._calculate_position_size(symbol, price)
        
        # Check if we have enough cash
        cost = abs(quantity) * price
        if cost > self.cash:
            logger.warning(f"Not enough cash ({self.cash:.2f}) for position costing {cost:.2f}")
            return None
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add metadata
        position.strategy = strategy_name
        position.metadata = metadata or {}
        
        # Update portfolio state
        self.positions[symbol] = position
        self.cash -= cost
        
        logger.info(f"Opened {position.quantity} {symbol} at {price:.2f}")
        
        # Save state if enabled
        if self.state_file:
            self.save_state(self.state_file)
        
        return position
    
    def close_position(
        self,
        symbol: str,
        price: float,
        date: Union[str, datetime],
        reason: str = "manual"
    ) -> Optional[Position]:
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close
            price: Exit price
            date: Exit date
            reason: Reason for closing
            
        Returns:
            Closed position if successful, None otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to close")
            return None
        
        position = self.positions[symbol]
        
        # Close the position
        position.close(price, date, reason)
        
        # Update portfolio state
        self.cash += abs(position.quantity) * price
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Closed {position.quantity} {symbol} at {price:.2f}, P&L: {position.pnl:.2f}")
        
        # Save state if enabled
        if self.state_file:
            self.save_state(self.state_file)
        
        return position
    
    def update_portfolio(
        self,
        prices: Dict[str, float],
        date: Union[str, datetime]
    ):
        """
        Update portfolio state with new prices.
        
        Args:
            prices: Dictionary of current prices (symbol -> price)
            date: Current date
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Calculate portfolio value
        portfolio_value = self.cash
        
        # Check for stop loss/take profit and update position values
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                
                # Check for stop loss/take profit
                should_close, reason = position.should_close(price)
                if should_close:
                    positions_to_close.append((symbol, price, reason))
                
                # Update portfolio value
                portfolio_value += position.quantity * price
        
        # Close positions that hit stops
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, date, reason)
        
        # Update equity curve and returns
        self.equity_curve[date] = portfolio_value
        
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve.iloc[-2]
            self.returns[date] = (portfolio_value / prev_value) - 1
        
        # Check portfolio stop loss
        if self.portfolio_stop_loss is not None:
            drawdown = (portfolio_value / self.initial_capital) - 1
            if drawdown <= -self.portfolio_stop_loss:
                logger.warning(f"Portfolio stop loss triggered! Drawdown: {-drawdown:.2%}")
                
                # Close all open positions
                for symbol in list(self.positions.keys()):
                    if symbol in prices:
                        self.close_position(symbol, prices[symbol], date, "portfolio_stop_loss")
        
        # Check portfolio target
        if self.portfolio_target is not None:
            gain = (portfolio_value / self.initial_capital) - 1
            if gain >= self.portfolio_target:
                logger.info(f"Portfolio target reached! Gain: {gain:.2%}")
        
        # Save state if enabled
        if self.state_file:
            self.save_state(self.state_file)
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on portfolio rules.
        
        Args:
            symbol: Symbol to calculate for
            price: Current price
            
        Returns:
            Number of shares/contracts to trade
        """
        # Maximum capital to allocate
        max_allocation = self.cash * self.max_capital_per_position
        
        # Calculate number of shares/contracts
        quantity = int(max_allocation / price)
        
        return quantity
    
    def _get_correlated_symbols(self, symbol: str) -> List[str]:
        """
        Get list of symbols correlated with the given symbol.
        
        Args:
            symbol: Symbol to check correlations for
            
        Returns:
            List of correlated symbols
        """
        if symbol not in self.correlation_matrix.index:
            return []
        
        correlations = self.correlation_matrix.loc[symbol]
        correlated = correlations[abs(correlations) >= self.correlation_threshold].index.tolist()
        
        # Remove the symbol itself
        if symbol in correlated:
            correlated.remove(symbol)
        
        return correlated
    
    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]):
        """
        Update correlation matrix with new price data.
        
        Args:
            price_data: Dictionary of price DataFrames (symbol -> DataFrame)
        """
        # Create a DataFrame with returns for each symbol
        returns_dict = {}
        
        for symbol, data in price_data.items():
            if 'close' in data.columns and len(data) > 1:
                returns_dict[symbol] = data['close'].pct_change().dropna()
        
        if len(returns_dict) > 1:
            returns_df = pd.DataFrame(returns_dict)
            self.correlation_matrix = returns_df.corr()
    
    def get_portfolio_value(self, prices: Dict[str, float] = None) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            prices: Optional dictionary of current prices (symbol -> price)
            
        Returns:
            Total portfolio value
        """
        if prices is None:
            # Use the most recent price available for each position
            return self.equity_curve.iloc[-1]
        
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                portfolio_value += position.quantity * prices[symbol]
        
        return portfolio_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        if len(self.equity_curve) < 2:
            return metrics
        
        # Total return
        metrics['total_return'] = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # Calculate daily returns
        daily_returns = self.equity_curve.pct_change().dropna()
        
        # Annualized return (assuming 252 trading days)
        n_days = len(daily_returns)
        n_years = n_days / 252
        metrics['annual_return'] = (1 + metrics['total_return']) ** (1 / max(n_years, 0.01)) - 1
        
        # Volatility (annualized)
        metrics['volatility'] = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Maximum drawdown
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Calmar ratio (return / max drawdown)
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Win rate for closed positions
        if len(self.closed_positions) > 0:
            winning_trades = sum(1 for p in self.closed_positions if p.pnl > 0)
            metrics['win_rate'] = winning_trades / len(self.closed_positions)
            
            # Average profit/loss
            metrics['avg_profit'] = sum(p.pnl for p in self.closed_positions if p.pnl > 0) / max(winning_trades, 1)
            metrics['avg_loss'] = sum(p.pnl for p in self.closed_positions if p.pnl <= 0) / max(len(self.closed_positions) - winning_trades, 1)
            
            # Profit factor
            total_profit = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
            total_loss = abs(sum(p.pnl for p in self.closed_positions if p.pnl <= 0))
            metrics['profit_factor'] = total_profit / max(total_loss, 0.01)
        
        return metrics
    
    def generate_report(self, save_dir: Optional[str] = None) -> str:
        """
        Generate a portfolio performance report.
        
        Args:
            save_dir: Optional directory to save report files
            
        Returns:
            Report summary as a string
        """
        # Get performance metrics
        metrics = self.get_performance_metrics()
        
        # Create report
        report = []
        report.append("===== PORTFOLIO REPORT =====")
        report.append(f"Initial Capital: ${self.initial_capital:.2f}")
        report.append(f"Current Value: ${self.equity_curve.iloc[-1]:.2f}")
        report.append(f"Available Cash: ${self.cash:.2f}")
        report.append(f"Open Positions: {len(self.positions)}")
        report.append(f"Closed Positions: {len(self.closed_positions)}")
        
        if metrics:
            report.append(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
            report.append(f"Annual Return: {metrics.get('annual_return', 0.0):.2%}")
            report.append(f"Volatility: {metrics.get('volatility', 0.0):.2%}")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
            report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
            report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0.0):.2f}")
            report.append(f"Win Rate: {metrics.get('win_rate', 0.0):.2%}")
            report.append(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
            report.append(f"Average Profit: ${metrics.get('avg_profit', 0.0):.2f}")
            report.append(f"Average Loss: ${metrics.get('avg_loss', 0.0):.2f}")
        
        # Open positions
        if self.positions:
            report.append("\nOpen Positions:")
            for symbol, position in self.positions.items():
                report.append(f"  {symbol}: {position.quantity} @ ${position.entry_price:.2f}")
        
        # Save to files if directory is provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            if metrics:
                pd.Series(metrics).to_csv(save_dir / 'metrics.csv')
            
            # Save equity curve
            self.equity_curve.to_csv(save_dir / 'equity_curve.csv')
            
            # Save positions
            open_positions = pd.DataFrame([p.to_dict() for p in self.positions.values()])
            if not open_positions.empty:
                open_positions.to_csv(save_dir / 'open_positions.csv', index=False)
            
            closed_positions = pd.DataFrame([p.to_dict() for p in self.closed_positions])
            if not closed_positions.empty:
                closed_positions.to_csv(save_dir / 'closed_positions.csv', index=False)
            
            # Save report
            with open(save_dir / 'report.txt', 'w') as f:
                f.write('\n'.join(report))
            
            # Create plots
            self.plot_equity_curve(save_path=str(save_dir / 'equity_curve.png'))
            self.plot_drawdown(save_path=str(save_dir / 'drawdown.png'))
            
            logger.info(f"Portfolio report saved to {save_dir}")
        
        return '\n'.join(report)
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Plot the equity curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.equity_curve) < 2:
            logger.warning("Not enough data to plot equity curve")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve.values)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Add horizontal line for initial capital
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """
        Plot the drawdown curve.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.equity_curve) < 2:
            logger.warning("Not enough data to plot drawdown")
            return
        
        daily_returns = self.equity_curve.pct_change().dropna()
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown.values * 100)
        plt.title('Portfolio Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Drawdown plot saved to {save_path}")
        else:
            plt.show()
    
    def save_state(self, filename: str):
        """
        Save portfolio state to a file.
        
        Args:
            filename: Path to save the state file
        """
        state = {
            'initial_capital': self.initial_capital,
            'capital': self.capital,
            'cash': self.cash,
            'positions': [p.to_dict() for p in self.positions.values()],
            'closed_positions': [p.to_dict() for p in self.closed_positions],
            'equity_curve': self.equity_curve.to_dict(),
            'returns': self.returns.to_dict(),
            'max_capital_per_position': self.max_capital_per_position,
            'max_positions': self.max_positions,
            'max_correlated_positions': self.max_correlated_positions,
            'correlation_threshold': self.correlation_threshold,
            'portfolio_stop_loss': self.portfolio_stop_loss,
            'portfolio_target': self.portfolio_target,
            'risk_free_rate': self.risk_free_rate
        }
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Portfolio state saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    def load_state(self, filename: str):
        """
        Load portfolio state from a file.
        
        Args:
            filename: Path to the state file
        """
        try:
            import json
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore basic properties
            self.initial_capital = state['initial_capital']
            self.capital = state['capital']
            self.cash = state['cash']
            self.max_capital_per_position = state['max_capital_per_position']
            self.max_positions = state['max_positions']
            self.max_correlated_positions = state['max_correlated_positions']
            self.correlation_threshold = state['correlation_threshold']
            self.portfolio_stop_loss = state['portfolio_stop_loss']
            self.portfolio_target = state['portfolio_target']
            self.risk_free_rate = state['risk_free_rate']
            
            # Restore positions
            self.positions = {}
            for pos_data in state['positions']:
                position = Position.from_dict(pos_data)
                self.positions[position.symbol] = position
            
            # Restore closed positions
            self.closed_positions = [Position.from_dict(p) for p in state['closed_positions']]
            
            # Restore equity curve and returns
            self.equity_curve = pd.Series(state['equity_curve'])
            self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
            
            self.returns = pd.Series(state['returns'])
            self.returns.index = pd.to_datetime(self.returns.index)
            
            logger.info(f"Portfolio state loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
