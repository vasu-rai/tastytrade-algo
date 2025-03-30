
# src/main.py
import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from tastytrade_algo.tastytrade_api import TastytradeAPI
from tastytrade_algo.strategy import MovingAverageCrossover

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    # Initialize API
    api = TastytradeAPI(
        username=os.getenv('TASTYTRADE_USERNAME'),
        password=os.getenv('TASTYTRADE_PASSWORD')
    )
    
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    
    # Main loop
    try:
        logger.info("Starting trading bot...")
        while True:
            # Here you would:
            # 1. Fetch market data
            # 2. Run strategy
            # 3. Execute trades if signals generated
            
            logger.info("Completed trading cycle")
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)

if __name__ == "__main__":
    main()
