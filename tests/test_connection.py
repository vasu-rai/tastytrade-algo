# test_connection.py
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to facilitate imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the API class
from src.tastytrade_algo.tastytrade_api import TastytradeAPI

# Load environment variables
load_dotenv()

def test_connection():
    # Debug output
    print(f"Username: {os.getenv('TASTYTRADE_USERNAME')}")
    print(f"Password length: {len(os.getenv('TASTYTRADE_PASSWORD')) if os.getenv('TASTYTRADE_PASSWORD') else 0}")
    
    # Initialize API
    api = TastytradeAPI(
        username=os.getenv('TASTYTRADE_USERNAME'),
        password=os.getenv('TASTYTRADE_PASSWORD')
    )
    
    # Test account info retrieval
    account_info = api.get_account_info()
    if account_info:
        print("Connection successful!")
        print(f"Number of accounts: {len(account_info)}")
    else:
        print("Connection failed")

if __name__ == "__main__":
    test_connection()
    