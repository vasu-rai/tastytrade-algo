# src/strategy.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Strategy:
    """Base strategy class"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        raise NotImplementedError("Subclasses must implement this method")

class MovingAverageCrossover(Strategy):
    """Moving average crossover strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"Initialized MA Crossover: short={short_window}, long={long_window}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover"""
        if len(data) < self.long_window:
            logger.warning(f"Insufficient data: {len(data)} points, need {self.long_window}")
            return data
            
        data = data.copy()
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals: 1 = buy, -1 = sell, 0 = hold
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
        
        # Get signal changes
        data['position_change'] = data['signal'].diff()
        
        logger.info(f"Generated signals: {sum(data['position_change'] != 0)} trade signals")
        return data
