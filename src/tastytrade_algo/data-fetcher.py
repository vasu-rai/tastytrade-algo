# src/tastytrade_algo/data_fetcher.py
import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetching and management for backtesting.
    
    This class handles downloading historical market data,
    caching it locally, and preparing it for backtesting.
    """
    
    def __init__(
        self,
        cache_dir: str = 'data/cache',
        api_key: Optional[str] = None,
        tastytrade_api = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded data
            api_key: Optional API key for data provider
            tastytrade_api: Optional TastytradeAPI instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.tastytrade_api = tastytrade_api
        self.session = requests.Session()
        
        logger.info(f"Initialized data fetcher with cache at {cache_dir}")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = 'daily',
        use_cache: bool = True,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime), defaults to today
            interval: Data interval ('daily', 'hourly', etc.)
            use_cache: Whether to use cached data
            force_download: Whether to force download even if cache exists
            
        Returns:
            DataFrame with OHLCV data
        """
        # Standardize dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check cache first
        cache_file = self._get_cache_filename(symbol, interval, start_date, end_date)
        
        if use_cache and not force_download and cache_file.exists():
            logger.info(f"Loading cached data for {symbol} from {cache_file}")
            return self._load_from_cache(cache_file)
        
        # If we need to download, determine which source to use
        if self.tastytrade_api is not None:
            logger.info(f"Downloading data for {symbol} using Tastytrade API")
            data = self._download_from_tastytrade(symbol, start_date, end_date, interval)
        else:
            logger.info(f"Downloading data for {symbol} using free API")
            data = self._download_from_free_api(symbol, start_date, end_date, interval)
        
        # Cache the downloaded data
        if use_cache and len(data) > 0:
            self._save_to_cache(data, cache_file)
        
        return data
    
    def _get_cache_filename(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Generate a cache filename based on parameters."""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return self.cache_dir / f"{symbol}_{interval}_{start_str}_{end_str}.csv"
    
    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load data from a cache file."""
        try:
            df = pd.read_csv(cache_file)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error loading cache file: {e}")
            return pd.DataFrame()
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path):
        """Save data to a cache file."""
        try:
            # Ensure the directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            data.to_csv(cache_file)
            logger.info(f"Data cached to {cache_file}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _download_from_tastytrade(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """
        Download data using the Tastytrade API.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.tastytrade_api is None or not hasattr(self.tastytrade_api, 'session'):
                logger.error("Tastytrade API not initialized properly")
                return pd.DataFrame()
            
            # Convert interval to API format
            interval_map = {
                'daily': 'day',
                'hourly': 'hour',
                'minute': 'minute'
            }
            api_interval = interval_map.get(interval, 'day')
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Construct API URL
            url = f"{self.tastytrade_api.base_url}/market-data/historical/equities/{symbol}/quotes"
            
            # Add query parameters
            params = {
                'start-date': start_str,
                'end-date': end_str,
                'timeframe': api_interval
            }
            
            # Make the request
            headers = self.tastytrade_api.session.headers
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            # Parse the response
            data = response.json()
            
            # Debug output
            logger.debug(f"API response structure: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            
            # Extract the data (structure may vary depending on the API)
            if 'data' in data and 'items' in data['data']:
                items = data['data']['items']
                
                # Convert to DataFrame
                df = pd.DataFrame(items)
                
                # Rename columns to standard OHLCV format if needed
                column_map = {
                    'time': 'date',
                    'open-price': 'open',
                    'high-price': 'high',
                    'low-price': 'low',
                    'close-price': 'close',
                    'volume': 'volume'
                }
                
                df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)
                
                # Convert types
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # Ensure all required columns exist
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in df.columns:
                        df[col] = np.nan
                
                return df
            
            logger.error("Unexpected API response format")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error downloading from Tastytrade API: {e}")
            return pd.DataFrame()
    
    def _download_from_free_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """
        Download data from a free API (fallback).
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Using Alpha Vantage as an example (you could use Yahoo Finance or other free sources)
            base_url = "https://www.alphavantage.co/query"
            
            # Convert interval to API format
            interval_map = {
                'daily': 'TIME_SERIES_DAILY',
                'weekly': 'TIME_SERIES_WEEKLY',
                'monthly': 'TIME_SERIES_MONTHLY'
            }
            function = interval_map.get(interval, 'TIME_SERIES_DAILY')
            
            # Create parameters
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            # Add API key if available
            if self.api_key:
                params['apikey'] = self.api_key
            
            # Make the request
            response = self.session.get(base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            # Parse the response
            data = response.json()
            
            # Extract time series data
            time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
            
            if time_series_key is None:
                logger.error(f"Unexpected API response format: {list(data.keys())}")
                if 'Note' in data:
                    logger.error(f"API Note: {data['Note']}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns to standard OHLCV format
            column_map = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            df.rename(columns=column_map, inplace=True)
            
            # Convert types
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Filter by date range
            df = df[(df.index >= pd.Timestamp(start_date)) & 
                   (df.index <= pd.Timestamp(end_date))]
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error downloading from free API: {e}")
            return pd.DataFrame()
            
    def prepare_data(
        self,
        data: pd.DataFrame,
        fillna: bool = True,
        dropna: bool = False
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting.
        
        Args:
            data: Raw OHLCV data
            fillna: Whether to fill NaN values
            dropna: Whether to drop rows with NaN values
            
        Returns:
            Cleaned DataFrame ready for backtesting
        """
        df = data.copy()
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Handle missing values
        if fillna:
            # Forward fill prices
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].fillna(method='ffill')
            
            # Fill remaining NaNs with zeros (e.g., for volume)
            df.fillna(0, inplace=True)
        elif dropna:
            df.dropna(inplace=True)
        
        # Ensure data is sorted by date
        if df.index.name == 'date':
            df = df.sort_index()
        
        # Add additional columns that might be useful
        if 'close' in df.columns:
            # Add returns
            df['returns'] = df['close'].pct_change()
            
            # Add log returns
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Add volatility (20-day rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def get_symbols_list(self, market: str = 'US') -> List[str]:
        """
        Get a list of available symbols.
        
        Args:
            market: Market to get symbols for (US, etc.)
            
        Returns:
            List of symbol strings
        """
        # This could be implemented with your specific data source
        # For now, return a placeholder
        if market == 'US':
            return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        return []
    
    def get_market_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary of DataFrames, keyed by symbol
        """
        result = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            try:
                data = self.get_historical_data(symbol, start_date, end_date, interval)
                if len(data) > 0:
                    result[symbol] = data
                    
                    # Add a small delay to avoid hitting rate limits
                    time.sleep(0.5)
                else:
                    logger.warning(f"No data returned for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return result