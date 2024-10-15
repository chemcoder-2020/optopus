from src.optopus.brokers.abstract_order import AbstractOptionOrder
from loguru import logger

class OptionBroker:
    def __init__(self, config):
        self.config = config
        self.order = self._connect_broker(self.config)
    
    def _connect_broker(self, config) -> AbstractOptionOrder:
        broker = config.get("broker", "Schwab")
        api_key = config.get("api_key")
        masked_api_key = "*" * (len(api_key) - 4) + api_key[-4:]
        client_secret = config.get("client_secret", None)
        redirect_uri = config.get("redirect_uri", None)
        token_file = config.get("token_file", "token.json")
        account_number = config.get("account_number", 0)

        logger.debug(f"Connecting to {broker} broker with API key {masked_api_key}...")
        if broker == "Schwab":
            from schwab_order import SchwabOptionOrder
            return SchwabOptionOrder(
                client_id=api_key,
                client_secret=client_secret,
                option_strategy=self.config.get("option_strategy"),
                redirect_uri=redirect_uri,
                token_file=token_file,
                which_account=account_number,
            )

