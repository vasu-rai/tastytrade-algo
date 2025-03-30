# src/tastytrade_algo/tastytrade_api.py
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TastytradeAPI:
    """Wrapper for Tastytrade API interactions"""
    
    def __init__(self, username: str, password: str):
        self.base_url = "https://api.tastyworks.com"  # Using production URL
        self.session = requests.Session()
        self.token = None
        self.login(username, password)
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate with Tastytrade API"""
        login_url = f"{self.base_url}/sessions"
        payload = {"login": username, "password": password}
        
        try:
            response = self.session.post(login_url, json=payload)
            response.raise_for_status()
            data = response.json()
            print(f"API Response: {data}")  # Debug output
            
            # Extract token from nested structure
            self.token = data["data"]["session-token"]
            # Use the proper header format (may need to be "Bearer token" or just the token)
            self.session.headers.update({"Authorization": self.token})
            logger.info("Successfully authenticated with Tastytrade API")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            print(f"Exception details: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return False 
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if not self.token:
            logger.error("Not authenticated")
            return None
            
        account_url = f"{self.base_url}/customers/me/accounts"
        try:
            response = self.session.get(account_url)
            response.raise_for_status()
            
            # Print response for debugging
            response_data = response.json()
            print(f"Account response: {response_data}")
            
            # The response might have data nested under a 'data' key
            if 'data' in response_data:
                return response_data['data']
            return response_data
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            print(f"Exception details: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return None