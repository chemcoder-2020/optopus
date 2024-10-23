from ..brokers.order import Order
from loguru import logger
import os


class OptionBroker:
    def __init__(self, config):
        self.config = config
        self.auth = self._get_auth_api()
        self.data = self._get_data_api()
        self.trading = self._get_trading_api()

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config}, broker={self.config.get('broker')})"

    def _get_auth_api(self):
        if self.config.get("broker", "Schwab").lower() == "schwab":
            from ..brokers.schwab.schwab_auth import SchwabAuth

            return SchwabAuth(
                client_id=self.config.get("api_key", os.getenv("SCHWAB_CLIENT_ID")),
                client_secret=self.config.get(
                    "client_secret", os.getenv("SCHWAB_CLIENT_SECRET")
                ),
                redirect_uri=self.config.get(
                    "redirect_uri",
                    os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1"),
                ),
                token_file=self.config.get(
                    "token_file", os.getenv("SCHWAB_TOKEN_FILE", "token.json")
                ),
            )

    def _get_data_api(self):
        if self.config.get("broker", "Schwab").lower() == "schwab":
            from ..brokers.schwab.schwab_data import SchwabData

            if self.auth is not None:
                return SchwabData(
                    client_id=self.config.get("api_key", os.getenv("SCHWAB_CLIENT_ID")),
                    client_secret=self.config.get(
                        "client_secret", os.getenv("SCHWAB_CLIENT_SECRET")
                    ),
                    redirect_uri=self.config.get(
                        "redirect_uri",
                        os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1"),
                    ),
                    token_file=self.config.get(
                        "token_file", os.getenv("SCHWAB_TOKEN_FILE", "token.json")
                    ),
                    auth=self.auth,
                )
            else:
                raise Exception(
                    "Authentication object is required to access Schwab Data API"
                )

    def _get_trading_api(self):
        if self.config.get("broker", "Schwab").lower() == "schwab":
            from ..brokers.schwab.schwab_trade import SchwabTrade

            if self.auth is not None:
                return SchwabTrade(
                    client_id=self.config.get("api_key", os.getenv("SCHWAB_CLIENT_ID")),
                    client_secret=self.config.get(
                        "client_secret", os.getenv("SCHWAB_CLIENT_SECRET")
                    ),
                    redirect_uri=self.config.get(
                        "redirect_uri",
                        os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1"),
                    ),
                    token_file=self.config.get(
                        "token_file", os.getenv("SCHWAB_TOKEN_FILE", "token.json")
                    ),
                    auth=self.auth,
                )
            else:
                raise Exception(
                    "Authentication object is required to access Schwab Trade API"
                )

    def update_config(self, config) -> None:
        """
        Update broker configuration and reinitialize API connections.
        
        Args:
            config: New configuration object
        """
        self.config = config
        self.auth = self._get_auth_api()
        self.data = self._get_data_api()
        self.trading = self._get_trading_api()

    def create_order(self, option_strategy) -> Order:
        broker = self.config.get("broker", "Schwab")
        api_key = self.config.get("api_key")
        client_secret = self.config.get("client_secret", None)
        redirect_uri = self.config.get("redirect_uri", None)
        token_file = self.config.get("token_file", "token.json")
        account_number = self.config.get("account_number", 0)

        logger.debug(f"Connecting to {broker} broker...")
        if broker.lower() == "schwab":
            from ..brokers.schwab.schwab_order import SchwabOptionOrder

            return SchwabOptionOrder(
                option_strategy=option_strategy,
                client_id=api_key if api_key else os.getenv("SCHWAB_CLIENT_ID"),
                client_secret=(
                    client_secret
                    if client_secret
                    else os.getenv("SCHWAB_CLIENT_SECRET")
                ),
                redirect_uri=(
                    redirect_uri
                    if redirect_uri
                    else os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
                ),
                token_file=(
                    token_file
                    if token_file
                    else os.getenv("SCHWAB_TOKEN_FILE", "token.json")
                ),
                which_account=account_number if account_number else 0,
            )
