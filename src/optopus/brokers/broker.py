from src.optopus.brokers.order import Order
from loguru import logger
import os


class OptionBroker:
    def __init__(self, config):
        self.config = config
        self.order = self.create_order(self.config)

    def create_order(self, config) -> Order:
        broker = config.get("broker", "Schwab")
        api_key = config.get("api_key")
        masked_api_key = "*" * (len(api_key) - 4) + api_key[-4:]
        client_secret = config.get("client_secret", None)
        redirect_uri = config.get("redirect_uri", None)
        token_file = config.get("token_file", "token.json")
        account_number = config.get("account_number", 0)

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
