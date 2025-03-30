# tests/test_strategy.py
import pandas as pd
import numpy as np
import pytest
from src.strategy import MovingAverageCrossover

def test_ma_crossover_signals():
    # Create sample data with a clear crossover
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 100, 50)])
    data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=10, long_window=30)
    
    # Generate signals
    result = strategy.generate_signals(data)
    
    # Check that we have both buy and sell signals
    assert 1 in result['signal'].values
    assert -1 in result['signal'].values
    
    # Check that position changes occur
    assert (result['position_change'] != 0).sum() > 0
    