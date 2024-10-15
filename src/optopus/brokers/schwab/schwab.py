from loguru import logger
import requests
import json
from .schwab_auth import SchwabAuth
import sys
import os


logger.add(
    sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
)


class Schwab:
    def __init__(
        self,
        client_id=os.getenv("SCHWAB_CLIENT_ID"),
        client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1"),
        token_file="token.json",
        auth=None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_file = token_file
        self.auth = (
            auth
            if auth
            else SchwabAuth(client_id, client_secret, redirect_uri, token_file)
        )
        self.trading_base_url = "https://api.schwabapi.com/trader/v1"

    def refresh_token(self):
        """
        Refresh the access token.

        Returns:
            bool: True if the access token was refreshed successfully, False otherwise.
        """
        try:
            _ = self.auth.refresh_access_token()
            self.auth.save_token(self.token_file)
            return True
        except Exception as e:
            logger.exception(f"Refreshing token failed: {e}")
            return False

    @logger.catch
    def _make_request(self, method, url, headers=None, params=None, data=None):
        """
        Make a general HTTP request.

        Args:
            method (str): The HTTP method (GET, POST, PUT, DELETE).
            url (str): The URL for the request.
            headers (dict): The headers for the request.
            params (dict): The query parameters for the request.
            data (dict): The data for the request.

        Returns:
            dict: The response data.
        """
        self.refresh_token()
        if not self.auth.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")

        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.auth.access_token}"
        headers["Accept"] = "application/json"

        response = requests.request(
            method, url, headers=headers, params=params, json=data
        )
        response.raise_for_status()
        return response

    def _get(self, url, params=None):
        return self._make_request("GET", url, params=params)

    def _post(self, url, data=None):
        return self._make_request("POST", url, data=data)

    def _put(self, url, data=None):
        return self._make_request("PUT", url, data=data)

    def _delete(self, url):
        return self._make_request("DELETE", url)

    def _load_token(self):
        """
        Load the access token from the token file.

        Returns:
            str: The access token.
        """
        with open(self.token_file, "r") as f:
            token_data = json.load(f)
            return token_data
