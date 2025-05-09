import httpx
import base64
from urllib.parse import urlparse, parse_qs
import json
import os
from loguru import logger


class SchwabAuth:
    def __init__(
        self, client_id=None, client_secret=None, redirect_uri=None, token_file=None
    ):
        self.client_id = client_id if client_id else os.getenv("SCHWAB_CLIENT_ID")
        self.client_secret = (
            client_secret if client_secret else os.getenv("SCHWAB_CLIENT_SECRET")
        )
        self.redirect_uri = (
            redirect_uri
            if redirect_uri
            else os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
        )
        self.access_token = None
        self.refresh_token = None
        self.token_data = None

        self.load_token()

    @property
    def token_file(self):
        return os.getenv("SCHWAB_TOKEN_FILE", "token.json")

    def load_token(self):
        """
        Load the access token from the token file.

        Returns:
            str: The access token.
        """
        if self.token_file and os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                token_data = json.load(f)
                self.access_token = token_data.get("access_token")
                self.refresh_token = token_data.get("refresh_token")
                self.token_data = token_data
                logger.info(f"Loaded token data from {self.token_file}")
                return self.token_data
        else:
            logger.info(f"Token file {self.token_file} not found. Trying to authenticate...")
            self.authenticate()
            

    def get_authorization_url(self):
        return f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self.client_id}&redirect_uri={self.redirect_uri}"

    def authenticate(self):
        import webbrowser

        authorization_url = self.get_authorization_url()
        logger.info(
            f"Please visit this URL to authorize the application: {authorization_url}"
        )
        webbrowser.open(authorization_url)

        authorization_url_response = input("Enter the authorization url response: ")
        tokens = self.get_tokens(authorization_url_response)
        logger.info("Access Token:", tokens["access_token"])
        logger.info("Refresh Token:", tokens["refresh_token"])
        self.save_token()

    def _get_token_url(self, authorization_url_response):
        # Parse the URL to extract the authorization code
        parsed_url = urlparse(authorization_url_response)
        query_params = parse_qs(parsed_url.query)

        # Extract the authorization code from the query parameters
        authorization_code = query_params.get("code", [None])[0]

        if authorization_code is None:
            raise ValueError("Authorization code not found in the URL")

        return authorization_code

    def get_tokens(self, authorization_url_response):
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        headers = {
            "Authorization": f'Basic {base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()}',
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "authorization_code",
            "code": self._get_token_url(authorization_url_response),
            "redirect_uri": self.redirect_uri,
        }

        response = httpx.post(token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        self.token_data = token_data
        return token_data

    def refresh_access_token(self):
        # self.load_token()  # try loading existing token file first
        try:
            url = "https://api.schwabapi.com/v1/oauth/token"
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
            response = httpx.post(url, headers=headers, data=data)
            response.raise_for_status()
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            self.token_data = tokens
        except Exception as e:
            self.load_token()  # try loading existing token file first
            url = "https://api.schwabapi.com/v1/oauth/token"
            headers = {
                "Authorization": f"Basic {base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
            response = httpx.post(url, headers=headers, data=data)
            response.raise_for_status()
            tokens = response.json()
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            self.token_data = tokens
        return tokens

    def save_token(self, path="token.json"):
        if self.token_data:
            if self.token_file:
                path = self.token_file
            with open(path, "w") as f:
                json.dump(self.token_data, f)

