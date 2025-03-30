# src/tastytrade_algo/data_fetcher.py
import os
import requests
import pandas as pd
import numpy as np
import logging
# Make sure date is imported
from datetime import datetime, timedelta, date, timezone
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
import json
# Import Polygon client
try:
    from polygon import RESTClient
    POLYGON_CLIENT_INSTALLED = True
except ImportError:
    POLYGON_CLIENT_INSTALLED = False
    # Define a dummy RESTClient if not installed, to prevent NameError
    # Users will get an error later if they try to use it without installation
    class RESTClient:
        def __init__(self, *args, **kwargs):
            logger.error("Polygon client library not installed. Run 'poetry add polygon-api-client'")
            raise ImportError("Polygon client library not installed.")


logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetching and management for backtesting.

    Handles downloading historical market data (equity and options),
    caching it locally, and preparing it for backtesting.
    """

    def __init__(
        self,
        cache_dir: str = 'data/cache',
        api_key: Optional[str] = None, # Alpha Vantage key
        tastytrade_api = None, # Original Tastytrade API class instance (if used)
        polygon_api_key: Optional[str] = None # ADDED: Polygon key parameter
    ):
        """
        Initialize the data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data
            api_key: Optional API key for Alpha Vantage
            tastytrade_api: Optional original TastytradeAPI instance (if used)
            polygon_api_key: Optional API key for Polygon.io
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key # Alpha Vantage key
        self.tastytrade_api = tastytrade_api
        self.polygon_api_key = polygon_api_key # Store Polygon key
        self.session = requests.Session() # For Alpha Vantage http requests

        logger.info(f"Initialized data fetcher with cache at {cache_dir}")
        # Log status of API keys
        if self.api_key:
            logger.info("Alpha Vantage API key provided.")
        else:
            logger.warning("Alpha Vantage API key not provided. Equity fallback may be limited.")
        if self.polygon_api_key:
            if POLYGON_CLIENT_INSTALLED:
                 logger.info("Polygon.io API key provided and client installed.")
            else:
                 logger.error("Polygon.io API key provided BUT client library 'polygon-api-client' is NOT installed.")
                 logger.error("Run: poetry add polygon-api-client")
        else:
            logger.warning("Polygon.io API key not provided. Options data fetching will be disabled.")
        if self.tastytrade_api:
             logger.info("Tastytrade API object provided.")
        else:
             logger.info("Tastytrade API object not provided.")


    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = 'daily',
        use_cache: bool = True,
        force_download: bool = False,
        instrument_type: str = 'equity' # ADDED: instrument type parameter
    ) -> pd.DataFrame:
        """
        Get historical price data for a symbol. (Updated docstring)

        Args:
            symbol: Ticker symbol (equity or option in Polygon format like O:SPY...)
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime), defaults to today
            interval: Data interval ('daily', 'hourly', 'minute', etc.)
            use_cache: Whether to use cached data
            force_download: Whether to force download even if cache exists
            instrument_type: Type of instrument ('equity' or 'option')

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Sanitize symbol for filename (replace common invalid chars)
        safe_symbol_part = symbol.replace(':','_').replace('/','_')
        cache_file = self._get_cache_filename(safe_symbol_part, interval, start_date, end_date)

        if use_cache and not force_download and cache_file.exists():
            logger.info(f"Loading cached data for {symbol} from {cache_file}")
            return self._load_from_cache(cache_file)

        data = pd.DataFrame()
        # --- MODIFIED: Logic to choose download method ---
        if instrument_type == 'option':
            if self.polygon_api_key and POLYGON_CLIENT_INSTALLED:
                logger.info(f"Downloading OPTIONS data for {symbol} using Polygon.io API")
                data = self._download_options_from_polygon(symbol, start_date, end_date, interval)
            elif not self.polygon_api_key:
                logger.error("Polygon API key not provided, cannot download options data.")
            else: # Key provided but client not installed
                 logger.error("Polygon client library not installed, cannot download options data. Run 'poetry add polygon-api-client'")
        elif instrument_type == 'equity':
            # Existing equity logic
            if self.tastytrade_api: # Check if the original API object exists
                 logger.info(f"Attempting to download EQUITY data for {symbol} using Tastytrade API")
                 data = self._download_from_tastytrade(symbol, start_date, end_date, interval)
                 if data.empty:
                     logger.warning("Tastytrade API download failed or returned no data, trying Alpha Vantage.")
                     data = self._download_from_free_api(symbol, start_date, end_date, interval)
            else:
                logger.info(f"Downloading EQUITY data for {symbol} using Alpha Vantage API")
                data = self._download_from_free_api(symbol, start_date, end_date, interval)
        else:
             logger.error(f"Unsupported instrument_type: {instrument_type}")
        # --- End logic ---

        if use_cache and not data.empty:
            self._save_to_cache(data, cache_file)

        return data

    def _get_cache_filename(
        self,
        symbol_part: str, # Changed param name for clarity
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Generate a cache filename based on parameters."""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        # Limit filename length if symbol_part is very long (like some option symbols)
        max_symbol_len = 50
        safe_symbol = symbol_part[:max_symbol_len] if len(symbol_part) > max_symbol_len else symbol_part
        return self.cache_dir / f"{safe_symbol}_{interval}_{start_str}_{end_str}.csv"

    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Load data from a cache file."""
        try:
            df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
            # Ensure index is UTC if it was saved with timezone
            if df.index.tz is None:
                 df.index = df.index.tz_localize('UTC')
            else:
                 df.index = df.index.tz_convert('UTC')
            return df
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            return pd.DataFrame()

    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path):
        """Save data to a cache file."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Save with index=True since 'date' is the index
            data.to_csv(cache_file, index=True)
            logger.info(f"Data cached to {cache_file}")
        except Exception as e:
            logger.error(f"Error saving to cache {cache_file}: {e}")

    # --- NEW METHOD ---
    def _download_options_from_polygon(
        self,
        option_symbol: str, # Expects Polygon format e.g., O:SPY251219C00500000
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """Downloads historical options aggregates from Polygon.io"""
        if not self.polygon_api_key or not POLYGON_CLIENT_INSTALLED:
            logger.error("Polygon API key or client library missing.")
            return pd.DataFrame()

        try:
            # Initialize RESTClient (consider initializing once in __init__ if frequently used)
            client = RESTClient(self.polygon_api_key)

            # Map interval to Polygon's timespan and multiplier
            multiplier = 1
            if interval == 'daily':
                timespan = 'day'
            elif interval == 'hourly':
                timespan = 'hour'
            elif interval == 'minute':
                timespan = 'minute'
            # Add more specific intervals (e.g., '5minute') by adjusting multiplier/timespan
            # Example: elif interval == '5minute': timespan = 'minute'; multiplier = 5
            else:
                logger.warning(f"Unsupported interval '{interval}' for Polygon, defaulting to 'day'.")
                timespan = 'day'

            # Format dates for Polygon API (YYYY-MM-DD string or date object)
            from_ = start_date.date() # Use date objects
            to = end_date.date()

            logger.info(f"Requesting Polygon Aggs for Options: {option_symbol}, {multiplier} {timespan}, from {from_} to {to}")

            # Use client.list_aggs which handles pagination automatically
            aggs_list = []
            # Note: Polygon uses 'from_' not 'from' due to Python keyword conflict
            # The client handles converting date objects to strings internally
            for bar in client.list_aggs(
                ticker=option_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_,
                to=to,
                limit=50000 # Max limit per request
            ):
                aggs_list.append(bar)

            if not aggs_list:
                logger.warning(f"Polygon API returned no aggregates for {option_symbol} in the specified range.")
                return pd.DataFrame()

            # Convert list of aggregate objects to DataFrame
            # Extract relevant fields directly
            records = []
            for bar in aggs_list:
                 records.append({
                     'timestamp': bar.timestamp,
                     'open': bar.open,
                     'high': bar.high,
                     'low': bar.low,
                     'close': bar.close,
                     'volume': bar.volume,
                     'vwap': getattr(bar, 'vwap', np.nan), # vwap might not always be present
                     'transactions': getattr(bar, 'transactions', np.nan) # transactions 'n' might not always be present
                 })
            df = pd.DataFrame(records)

            # Convert timestamp (milliseconds) to datetime index (UTC)
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('date', inplace=True)
                df.drop(columns=['timestamp'], inplace=True)
            else:
                logger.error("Timestamp column 't' missing in Polygon response processing.")
                return pd.DataFrame()

            # Select and ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            # Add optional ones if they exist
            optional_cols = ['vwap', 'transactions']
            final_cols = required_cols + [col for col in optional_cols if col in df.columns]

            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan # Add missing required columns if needed

            df = df[final_cols] # Keep selected columns

            # Ensure numeric types (float for prices/vwap, Int64 for volume/transactions)
            for col in ['open', 'high', 'low', 'close', 'vwap']:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce')
            for col in ['volume', 'transactions']:
                 if col in df.columns:
                      # Use Int64 which supports NaN, unlike int
                      df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')


            df.sort_index(inplace=True)

            logger.info(f"Successfully fetched and processed {len(df)} records for {option_symbol} from Polygon.io")
            return df

        except Exception as e:
            logger.error(f"Error downloading/processing data from Polygon.io API for {option_symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    # --- Keep existing methods below ---
    def _download_from_tastytrade(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """ (Keep original implementation or SDK version if you were working on it) """
        logger.warning("_download_from_tastytrade called, but may not be fully implemented or functional for historical data.")
        # Keep your previous implementation here if desired, otherwise return empty
        return pd.DataFrame()


    def _download_from_free_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """ (Keep the updated Alpha Vantage implementation) """
        try:
            # --- Keep the improved Alpha Vantage code from previous step ---
            if not self.api_key:
                logger.error("Alpha Vantage API key is required for downloading equity data.")
                return pd.DataFrame()

            base_url = "https://www.alphavantage.co/query"
            interval_map = {
                'daily': 'TIME_SERIES_DAILY',
                'weekly': 'TIME_SERIES_WEEKLY',
                'monthly': 'TIME_SERIES_MONTHLY'
            }
            function = interval_map.get(interval, 'TIME_SERIES_DAILY')

            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': 'full',
                'datatype': 'json',
                'apikey': self.api_key
            }

            logger.info(f"Requesting Alpha Vantage: {params}")
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return pd.DataFrame()
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API Note: {data['Note']}")

            time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
            if time_series_key is None:
                logger.error(f"Unexpected Alpha Vantage API response format: {list(data.keys())}")
                return pd.DataFrame()

            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            column_map = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}
            df.rename(columns=column_map, inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                else:
                    df[col] = np.nan

            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            df.sort_index(inplace=True)
            logger.info(f"Successfully fetched {len(df)} records for {symbol} from Alpha Vantage")
            return df
        except requests.exceptions.RequestException as req_err:
            logger.error(f"HTTP Error fetching data from Alpha Vantage for {symbol}: {req_err}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()


    def prepare_data(
        self,
        data: pd.DataFrame,
        fillna: bool = True,
        dropna: bool = False
    ) -> pd.DataFrame:
        """ (Keep original implementation, ensure ffill is used correctly) """
        df = data.copy()
        required_cols = ['open', 'high', 'low', 'close'] # Volume not strictly required for all calcs
        missing_cols = [col for col in required_cols if col not in df.columns or df[col].isnull().all()]

        if missing_cols:
            # Log warning instead of error if only volume is missing but others are present
            if set(missing_cols) == {'volume'}:
                 logger.warning(f"Column 'volume' is missing or all NaN for symbol.")
            elif len(missing_cols) < len(required_cols) + 1: # Allow volume to be missing
                 logger.warning(f"Potentially missing required columns or all NaN: {missing_cols}")
            else:
                 logger.error(f"Missing required price columns or all NaN: {missing_cols}")
                 return pd.DataFrame()


        # Handle missing values
        if fillna:
            # Forward fill prices
            for col in ['open', 'high', 'low', 'close']:
                 if col in df.columns:
                      # Use recommended ffill()
                      df[col] = df[col].ffill()

            # Fill volume NAs with 0 AFTER ffill on prices
            if 'volume' in df.columns:
                 df['volume'] = df['volume'].fillna(0)

            # If any price columns still have NaN at the beginning after ffill, drop those rows
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        elif dropna:
            df.dropna(inplace=True)

        # Ensure data is sorted by date
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Add additional columns if 'close' exists and is not all NaN
        if 'close' in df.columns and not df['close'].isnull().all():
            df['returns'] = df['close'].pct_change()
            # Use np.log1p on the returns for numerical stability if returns can be -1
            # Or handle potential zeros/negatives in close prices if calculating log(close/close.shift)
            with np.errstate(divide='ignore', invalid='ignore'): # Suppress log(0) warnings
                 df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                 df['log_returns'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero

            df['volatility'] = df['returns'].rolling(window=20).std()

        return df

    def get_symbols_list(self, market: str = 'US') -> List[str]:
        """ (Keep original implementation) """
        if market == 'US':
            return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        return []

    def get_market_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = 'daily',
        # ADDED: Allow passing instrument type for all symbols in the list
        instrument_type: str = 'equity'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols. (Updated docstring)

        Args:
            symbols: List of ticker symbols (equity or option in Polygon format)
            start_date: Start date
            end_date: End date
            interval: Data interval
            instrument_type: Type of instrument for ALL symbols in list ('equity' or 'option')

        Returns:
            Dictionary of DataFrames, keyed by symbol
        """
        result = {}
        delay_between_calls = 0.5 # Default delay

        # Polygon free tier allows 5 calls/min -> 12 seconds per call
        if instrument_type == 'option' and self.polygon_api_key:
             # Check if key is likely free tier - adjust delay
             # This is a guess; paid keys have higher limits.
             # A more robust solution would check subscription level if possible.
             if 'free' in self.polygon_api_key.lower() or len(self.polygon_api_key) < 20: # Heuristic
                  delay_between_calls = 13 # Be safe with free tier limit
             else:
                  delay_between_calls = 0.2 # Faster for likely paid keys

        for symbol in symbols:
            logger.info(f"Fetching data for {symbol} (Type: {instrument_type})")
            try:
                # Pass instrument_type to get_historical_data
                data = self.get_historical_data(
                    symbol,
                    start_date,
                    end_date,
                    interval,
                    instrument_type=instrument_type # Pass type here
                )
                if data is not None and not data.empty:
                    result[symbol] = data
                else:
                    logger.warning(f"No data returned or processed for {symbol}")

                # Add delay to avoid hitting rate limits
                logger.debug(f"Waiting {delay_between_calls}s before next API call...")
                time.sleep(delay_between_calls)

            except Exception as e:
                logger.error(f"Error in get_market_data loop for {symbol}: {e}", exc_info=True)

        return result