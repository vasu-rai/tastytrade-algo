import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple mocks for the missing tastytrade-algo classes
class MockSession:
    def __init__(self, username=None, password=None, paper_trading=True):
        self.username = username
        self.paper_trading = paper_trading
        print(f"Initialized mock session for {username} (Paper trading: {paper_trading})")
    
    def get_account_info(self):
        return {"account_number": "DEMO123", "buying_power": 25000, "cash_balance": 10000}
    
    def get_positions(self):
        return [{"symbol": "SPY", "quantity": 100, "avg_price": 420.50}]
    
    def place_order(self, order):
        return {"status": "filled", "order_id": "mock-order-123"}

class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def get_price(self):
        # Mock current price
        base_prices = {"SPY": 428.75, "AAPL": 175.50, "MSFT": 405.25, "NVDA": 875.80}
        return base_prices.get(self.symbol, 100 + np.random.random() * 10)

class Option:
    def __init__(self, symbol, strike, expiration, option_type):
        self.symbol = symbol
        self.strike = strike
        self.expiration = expiration
        self.option_type = option_type
    
    def __str__(self):
        return f"{self.symbol} {self.expiration} {self.strike} {self.option_type}"
    
    def get_price(self):
        # Mock option price
        base_price = Stock(self.symbol).get_price()
        if self.option_type == 'call':
            return max(0.10, round(abs(base_price - self.strike) * 0.1, 2))
        else:
            return max(0.10, round(abs(self.strike - base_price) * 0.1, 2))

class Order:
    def __init__(self, instrument, quantity, order_type, price, side):
        self.instrument = instrument
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.side = side

def get_option_chains(symbol):
    # Mock option chain data
    base_price = Stock(symbol).get_price()
    
    today = datetime.now()
    expirations = [
        (today + timedelta(days=7)).strftime('%Y-%m-%d'),
        (today + timedelta(days=30)).strftime('%Y-%m-%d'),
        (today + timedelta(days=60)).strftime('%Y-%m-%d'),
        (today + timedelta(days=90)).strftime('%Y-%m-%d')
    ]
    
    chains = {}
    for exp in expirations:
        chains[exp] = []
        # Add strikes around the current price
        for strike in range(int(base_price * 0.8), int(base_price * 1.2) + 1, 5):
            chains[exp].append(Option(symbol, strike, exp, 'call'))
            chains[exp].append(Option(symbol, strike, exp, 'put'))
    
    return chains

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trading Algorithm Example')
    parser.add_argument('--example', type=str, default='usage',
                        help='Example to run: usage, backtest')
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Symbol to use for the example')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtest (format: YYYY-MM-DD)')
    return parser.parse_args()

def example_usage(symbol='SPY'):
    """
    Basic example of using mock trading functionality
    """
    print(f"Running basic usage example for {symbol}")
    
    # Initialize the mock session
    session = MockSession(
        username=os.environ.get('USERNAME', 'demo_user'),
        paper_trading=True  # Use paper trading account
    )
    
    # Get account info
    account = session.get_account_info()
    print(f"Account: {account}")
    
    # Get positions
    positions = session.get_positions()
    print(f"Current positions: {positions}")
    
    # Get instrument data
    stock = Stock(symbol)
    price = stock.get_price()
    print(f"{symbol} current price: ${price:.2f}")
    
    # Get option chains
    chains = get_option_chains(symbol)
    print(f"Found {len(chains)} option expirations")
    
    # Example: Get ATM call option 30-45 days out
    today = datetime.now()
    target_date = today + timedelta(days=30)
    
    # Find the closest expiration date
    expirations = sorted(chains.keys())
    closest_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
    print(f"Selected expiration: {closest_exp}")
    
    # Get ATM call option
    calls = [opt for opt in chains[closest_exp] if opt.option_type == 'call']
    atm_call = min(calls, key=lambda x: abs(x.strike - price))
    print(f"Selected ATM call: {atm_call}")
    
    # Example trade (real order not sent)
    order = Order(
        instrument=atm_call,
        quantity=1,
        order_type='limit',
        price=atm_call.get_price() * 1.05,  # 5% above current price
        side='buy'
    )
    print(f"Example order: Buy 1 {atm_call} at ${order.price:.2f}")
    print("Order not sent (mock mode)")
    
    print("Basic example completed")

def run_backtest(symbol='SPY', start_date=None, end_date=None):
    """
    Example of running a simple backtest
    """
    print(f"Running backtest for {symbol}")
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Backtest period: {start_date} to {end_date}")
    
    try:
        # Create simulated historical data
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate business days (excluding weekends)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        # Start with a base price
        base_prices = {"SPY": 420, "AAPL": 170, "MSFT": 400, "NVDA": 850}
        start_price = base_prices.get(symbol, 100)
        
        # Generate random price movement with slight upward bias
        prices = [start_price]
        for i in range(1, len(date_range)):
            # Random daily change between -2% and +2% with slight upward bias
            daily_change = np.random.normal(0.0005, 0.015)  
            prices.append(prices[-1] * (1 + daily_change))
        
        # Create DataFrame
        historical_data = pd.DataFrame({
            'date': date_range,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, size=len(date_range))
        })
        
        # Simple moving average strategy
        historical_data['sma_20'] = historical_data['close'].rolling(window=20).mean()
        historical_data['sma_50'] = historical_data['close'].rolling(window=50).mean()
        
        # Generate signals
        historical_data['signal'] = 0
        historical_data.loc[historical_data['sma_20'] > historical_data['sma_50'], 'signal'] = 1
        historical_data.loc[historical_data['sma_20'] < historical_data['sma_50'], 'signal'] = -1
        
        # Calculate returns
        historical_data['returns'] = historical_data['close'].pct_change()
        historical_data['strategy_returns'] = historical_data['signal'].shift(1) * historical_data['returns']
        
        # Calculate cumulative returns
        historical_data['cum_returns'] = (1 + historical_data['returns']).cumprod()
        historical_data['strategy_cum_returns'] = (1 + historical_data['strategy_returns']).cumprod()
        
        # Calculate strategy metrics
        strategy_return = (historical_data['strategy_cum_returns'].iloc[-1] - 1) * 100
        buy_hold_return = (historical_data['cum_returns'].iloc[-1] - 1) * 100
        
        # Calculate max drawdown
        historical_data['strategy_peak'] = historical_data['strategy_cum_returns'].cummax()
        historical_data['strategy_drawdown'] = (historical_data['strategy_cum_returns'] / historical_data['strategy_peak'] - 1) * 100
        max_drawdown = historical_data['strategy_drawdown'].min()
        
        # Print results
        print("\nBacktest Results:")
        print(f"Start Price: ${historical_data['close'].iloc[0]:.2f}")
        print(f"End Price: ${historical_data['close'].iloc[-1]:.2f}")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Strategy Return: {strategy_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {(252**0.5) * (historical_data['strategy_returns'].mean() / historical_data['strategy_returns'].std()):.2f}")
        
        # Calculate trade statistics
        historical_data['position_change'] = historical_data['signal'].diff().fillna(0)
        trades = historical_data[historical_data['position_change'] != 0]
        
        print(f"Total Trades: {len(trades)}")
        
        # Plotting
        plt.figure(figsize=(12, 10))
        
        # Price and moving averages
        plt.subplot(3, 1, 1)
        plt.plot(historical_data['date'], historical_data['close'], label='Price')
        plt.plot(historical_data['date'], historical_data['sma_20'], label='SMA 20')
        plt.plot(historical_data['date'], historical_data['sma_50'], label='SMA 50')
        plt.title(f'{symbol} Price and Moving Averages')
        plt.legend()
        
        # Buy/Sell signals
        plt.subplot(3, 1, 2)
        plt.plot(historical_data['date'], historical_data['close'], alpha=0.3, color='gray')
        
        # Plot buy signals
        buy_signals = historical_data[historical_data['position_change'] > 0]
        plt.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='green', s=100, label='Buy')
        
        # Plot sell signals
        sell_signals = historical_data[historical_data['position_change'] < 0]
        plt.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='red', s=100, label='Sell')
        
        plt.title(f'{symbol} Trading Signals')
        plt.legend()
        
        # Strategy performance
        plt.subplot(3, 1, 3)
        plt.plot(historical_data['date'], historical_data['cum_returns'], label='Buy & Hold')
        plt.plot(historical_data['date'], historical_data['strategy_cum_returns'], label='Strategy')
        plt.title('Strategy Performance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{symbol}_backtest_results.png")
        print(f"Backtest chart saved as {symbol}_backtest_results.png")
        
        return historical_data
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.example == 'usage':
        example_usage(args.symbol)
    elif args.example == 'backtest':
        run_backtest(args.symbol, args.start_date, args.end_date)
    else:
        print(f"Unknown example: {args.example}")
        print("Available examples: usage, backtest")