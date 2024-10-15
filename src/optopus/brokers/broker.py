from ..brokers.order import Order
from loguru import logger
import os


class OptionBroker:
    def __init__(self, config):
        self.config = config
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config}, broker={self.config.get('broker')})"

    def _get_data_api(self):
        if self.config.get("broker", "Schwab") == "Schwab":
            from src.optopus.brokers.schwab_data import SchwabData
            return SchwabData(
                client_id=self.config.get("api_key"),
                client_secret=self.config.get("client_secret"),
                redirect_uri=self.config.get("redirect_uri"),
                token_file=self.config.get("token_file", "token.json"),
            )
        

    def create_order(self) -> Order:
        broker = self.config.get("broker", "Schwab")
        api_key = self.config.get("api_key")
        masked_api_key = "*" * (len(api_key) - 4) + api_key[-4:]
        client_secret = self.config.get("client_secret", None)
        redirect_uri = self.config.get("redirect_uri", None)
        token_file = self.config.get("token_file", "token.json")
        account_number = self.config.get("account_number", 0)

        logger.debug(f"Connecting to {broker} broker with API key {masked_api_key}...")
        if broker == "Schwab":
            from src.optopus.brokers.schwab_order import SchwabOptionOrder

            return SchwabOptionOrder(
                option_strategy=self.config.get("option_strategy"),
                client_id=api_key if api_key else os.getenv("SCHWAB_CLIENT_ID"),
                client_secret=(
                    client_secret
                    if client_secret
                    else os.getenv("SCHWAB_CLIENT_SECRET")
                ),
                redirect_uri=(
                    redirect_uri if redirect_uri else os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
                ),
                token_file=(
                    token_file
                    if token_file
                    else os.getenv("SCHWAB_TOKEN_FILE", "token.json")
                ),
                which_account=account_number if account_number else 0,
            )
