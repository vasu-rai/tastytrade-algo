# Tastytrade Algorithmic Trading System

A comprehensive quantitative trading system designed to work with the Tastytrade API. This system provides a framework for backtesting, optimizing, and deploying trading strategies with institutional-quality features.

## Features

- **Backtesting Engine**: Test trading strategies against historical data with realistic trading simulation
- **Strategy Framework**: Create and customize trading strategies with a flexible architecture
- **Portfolio Management**: Manage positions, risk, and capital allocation across multiple strategies
- **Data Management**: Fetch, clean, and prepare market data for analysis and trading
- **Performance Analytics**: Track and analyze trading performance with comprehensive metrics
- **Command Line Interface**: Control the system through an easy-to-use CLI
- **Tastytrade API Integration**: Connect to the Tastytrade platform for live trading

## Installation

1. Clone the repository and navigate to the root directory:

```bash
git clone https://github.com/your-username/tastytrade-algo.git
cd tastytrade-algo
```

2. Set up a Python virtual environment using Poetry:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

1. Create a `.env` file in the root directory with your Tastytrade credentials:

```
TASTYTRADE_USERNAME=your_username
TASTYTRADE_PASSWORD=your_password
```

## Directory Structure

```
tastytrade-algo/
├── config/            # Configuration files
├── data/              # Market data storage
│   └── cache/         # Cached data files
├── examples/          # Example scripts
├── output/            # Output files and reports
├── src/               # Source code
│   └── tastytrade_algo/
│       ├── backtest.py        # Backtesting engine
│       ├── cli.py             # Command line interface
│       ├── data_fetcher.py    # Data management
│       ├── portfolio.py       # Portfolio management
│       ├── strategy.py        # Strategy framework
│       ├── strategy_manager.py # Strategy management
│       ├── tastytrade_api.py  # API wrapper
│       └── trading_engine.py  # Main trading engine
├── tests/             # Test files
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

## Quick Start

### Basic Backtest

Run a simple backtest using the Moving Average Crossover strategy:

```bash
poetry run python examples/simple_backtest.py --example backtest
```

### Parameter Optimization

Optimize the parameters of a strategy:

```bash
poetry run python examples/simple_backtest.py --example optimize
```

### Multi-Strategy Backtest

Run a backtest with multiple strategies and custom allocations:

```bash
poetry run python examples/simple_backtest.py --example multi_strategy
```

### Paper Trading Simulation

Run a paper trading simulation for 5 minutes:

```bash
poetry run python examples/simple_backtest.py --example paper
```

## Using the CLI

The system provides a command-line interface for all operations:

### Running a Backtest

```bash
poetry run python -m src.tastytrade_algo.cli backtest --start-date 2020-01-01 --end-date 2023-12-31 --symbols SPY QQQ
```

### Creating a Strategy

```bash
poetry run python -m src.tastytrade_algo.cli create-strategy --name ma_crossover --type MovingAverageCrossover --symbols SPY QQQ --params '{"short_window": 20, "long_window": 50}'
```

### Optimizing a Strategy

```bash
poetry run python -m src.tastytrade_algo.cli optimize --strategy ma_crossover --symbols SPY --param-ranges '{"short_window": [10, 20, 30], "long_window": [50, 100, 150]}'
```

### Running Paper Trading

```bash
poetry run python -m src.tastytrade_algo.cli paper --interval 30
```

### Getting System Status

```bash
poetry run python -m src.tastytrade_algo.cli status
```

### Generating Reports

```bash
poetry run python -m src.tastytrade_algo.cli report --type portfolio
```

## Creating Custom Strategies

To create a custom strategy, extend the base `Strategy` class:

```python
from src.tastytrade_algo.strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        # Your strategy logic here
        data = data.copy()
        
        # Calculate indicators
        # ...
        
        # Generate signals (1 = buy, -1 = sell, 0 = hold)
        data['signal'] = 0
        # ...
        
        # Calculate position changes
        data['position_change'] = data['signal'].diff()
        
        return data
```

## Trading Engine Configuration

You can configure the trading engine using a JSON file:

```json
{
  "mode": "backtest",
  "data_dir": "data",
  "output_dir": "output",
  "api": {
    "username": "your_username",
    "password": "your_password",
    "environment": "prod"
  },
  "portfolio": {
    "initial_capital": 100000.0,
    "max_capital_per_position": 0.1,
    "max_positions": 10,
    "portfolio_stop_loss": 0.2
  },
  "backtest": {
    "commission": 0.0,
    "slippage": 0.001
  }
}
```

## Advanced Usage

### Portfolio Management

The system includes sophisticated portfolio management capabilities:

- Position sizing based on volatility and risk
- Maximum drawdown protection
- Correlation-based position limits
- Performance-based capital allocation

### Multi-Strategy Approach

You can run multiple strategies simultaneously with different allocation methods:

- Equal allocation
- Performance-based allocation
- Custom allocation

## License

This project is licensed under the terms of the proprietary license for Ceryneian Partners LLC.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
