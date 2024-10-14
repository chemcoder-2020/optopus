import requests
import base64
from urllib.parse import urlparse, parse_qs
import json
import os
import logging

logging.basicConfig(level=logging.INFO)


class SchwabAuth:
    def __init__(self, client_id, client_secret, redirect_uri, token_file=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_file = token_file
        self.access_token = None
        self.refresh_token = None
        self.token_data = None
        self.logger = logging.getLogger(__name__)

        if token_file and os.path.exists(token_file):
            with open(token_file, "r") as f:
                token_data = json.load(f)
                self.access_token = token_data.get("access_token")
                self.refresh_token = token_data.get("refresh_token")
                self.token_data = token_data

            self.logger.info(f"Loaded token data from {token_file}")


    def get_authorization_url(self):
        return f"https://api.schwabapi.com/v1/oauth/authorize?client_id={self.client_id}&redirect_uri={self.redirect_uri}"

    def authenticate(self):
        import webbrowser

        authorization_url = self.get_authorization_url()
        print(
            f"Please visit this URL to authorize the application: {authorization_url}"
        )
        webbrowser.open(authorization_url)

        authorization_url_response = input("Enter the authorization url response: ")
        tokens = self.get_tokens(authorization_url_response)
        print("Access Token:", tokens["access_token"])
        print("Refresh Token:", tokens["refresh_token"])
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

        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        self.token_data = token_data
        return token_data

    def refresh_access_token(self):
        url = "https://api.schwabapi.com/v1/oauth/token"
        headers = {
            "Authorization": f"Basic {base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        tokens = response.json()
        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]
        self.token_data = tokens
        return tokens

    def save_token(self, path="token.json"):
        if self.token_data:
            with open(path, "w") as f:
                json.dump(self.token_data, f)


if __name__ == "__main__":
    client_id = "WjGR3xmyXiUGGdewERBkiJkmVYLWKsM4"
    client_secret = "eu5aIVUNx7ReRK2f"
    redirect_uri = "https://127.0.0.1"
    token_file = "token.json"
    auth = SchwabAuth(client_id, client_secret, redirect_uri, token_file="token.json")
    auth.authenticate()
