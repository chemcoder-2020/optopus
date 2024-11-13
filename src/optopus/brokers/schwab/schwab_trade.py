import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Union
from loguru import logger
import dotenv
from .schwab import Schwab


dotenv.load_dotenv()


class SchwabTrade(Schwab):
    def __init__(
        self,
        client_id,
        client_secret,
        redirect_uri="https://127.0.0.1",
        token_file="token.json",
        auth=None,
    ):
        """
        Initialize the SchwabTrade class with the client ID, client secret, redirect URI, token file path, and optional auth instance.

        Args:
            client_id (str): The client ID.
            client_secret (str): The client secret.
            redirect_uri (str): The redirect URI.
            token_file (str): The path to the token file.
            auth (SchwabAuth, optional): An optional auth instance. Defaults to None.
        """
        super().__init__(client_id, client_secret, redirect_uri, token_file, auth)

        try:
            self.auth.refresh_access_token()
        except Exception as e:
            logger.info(
                f"{e}... Attempting to get new token by going through the authentication process"
            )
            self.auth.authenticate()

        self.trading_base_url = "https://api.schwabapi.com/trader/v1"

    def get_account_numbers(self):
        url = f"{self.trading_base_url}/accounts/accountNumbers"
        response = self._get(url)
        self.account_numbers = response.json()
        return self.account_numbers

    def get_all_account_info(self, positions=True):
        url = f"{self.trading_base_url}/accounts"
        params = None
        if positions:
            params = {"fields": "positions"}
        response = self._get(url, params=params)
        return response.json()

    def get_specific_account_info(self, account_hashValue, positions=True):
        url = f"{self.trading_base_url}/accounts/{account_hashValue}"
        params = None
        if positions:
            params = {"fields": "positions"}
        response = self._get(url, params=params)
        return response.json()

    def get_orders(
        self,
        account_number_hash_value,
        max_results=3000,
        from_entered_time=None,
        to_entered_time=None,
        status=None,
    ):
        """
        Get orders for a specified account.

        Args:
            account_number_hash_value (str): The encrypted ID of the account.
            max_results (int): The max number of orders to retrieve. Default is 3000.
            from_entered_time (str): Specifies that no orders entered before this time should be returned. Valid ISO-8601 format.
            to_entered_time (str): Specifies that no orders entered after this time should be returned. Valid ISO-8601 format.
            status (str): Specifies that only orders of this status should be returned.

        Returns:
            dict: The response data.
        """
        url = f"{self.trading_base_url}/accounts/{account_number_hash_value}/orders"
        params = {
            "maxResults": max_results,
        }

        # Default from_entered_time to one month ago from today
        if from_entered_time is None:
            from_entered_time = (
                datetime.utcnow() - timedelta(days=30)
            ).isoformat() + "Z"
        else:
            from_entered_time = from_entered_time.isoformat() + "Z"

        # Default to_entered_time to today
        if to_entered_time is None:
            to_entered_time = datetime.utcnow().isoformat() + "Z"
        else:
            to_entered_time = to_entered_time.isoformat() + "Z"

        params["fromEnteredTime"] = from_entered_time
        params["toEnteredTime"] = to_entered_time

        if status:
            params["status"] = status

        return self._get(url, params=params).json()

    @classmethod
    def generate_single_option_json(
        cls,
        symbol: str,
        expiration: Union[str, pd.Timestamp],
        option_type: str,
        strike_price: float,
        instruction: str,
        quantity: int,
        order_type: str,
        price: float,
        duration: str,
    ) -> dict:
        """
        Generate JSON for a single option trade.

        Args:
            symbol (str): The underlying symbol.
            expiration (Union[str, pd.Timestamp]): The expiration date in YYMMDD format or as a pd.Timestamp.
            option_type (str): The type of option ('C' for call, 'P' for put).
            strike_price (float): The strike price.
            instruction (str): The instruction (e.g., 'BUY_TO_OPEN', 'SELL_TO_OPEN').
            quantity (int): The number of contracts.
            order_type (str): The type of order (e.g., 'LIMIT', 'MARKET').
            price (float): The price for limit orders.
            duration (str): The duration of the order (e.g., 'DAY', 'GOOD_TILL_CANCEL').

        Returns:
            dict: The JSON payload for the trade.
        """
        if option_type not in ["C", "P"]:
            raise ValueError(
                f"Invalid option type: {option_type}. Only 'C' (call) and 'P' (put) are supported."
            )

        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.strftime("%y%m%d")
        elif isinstance(expiration, str):
            if len(expiration) != 6:
                expiration = pd.Timestamp(expiration).strftime("%y%m%d")

        option_symbol = f"{symbol.ljust(6)}{expiration}{option_type}{str(int(strike_price * 1000)).zfill(8)}"
        payload = {
            "complexOrderStrategyType": "NONE",
            "orderType": order_type,
            "session": "NORMAL",
            "price": f"{price:.2f}",
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {"symbol": option_symbol, "assetType": "OPTION"},
                }
            ],
        }
        return payload

    @classmethod
    def generate_vertical_spread_json(
        cls,
        symbol,
        expiration,
        long_option_type,
        long_strike_price,
        short_option_type,
        short_strike_price,
        quantity,
        price,
        duration,
        is_entry=True,
    ):
        """
        Generate JSON for a vertical spread trade.

        Args:
            symbol (str): The underlying symbol.
            expiration (str): The expiration date in YYMMDD format.
            long_option_type (str): The type of long option ('C' for call, 'P' for put).
            long_strike_price (float): The long strike price.
            short_option_type (str): The type of short option ('C' for call, 'P' for put).
            short_strike_price (float): The short strike price.
            quantity (int): The number of contracts.
            order_type (str): The type of order (e.g., 'NET_DEBIT', 'NET_CREDIT').
            price (float): The price for the spread.
            duration (str): The duration of the order (e.g., 'DAY', 'GOOD_TILL_CANCEL').

        Returns:
            dict: The JSON payload for the trade.
        """
        if long_option_type not in ["C", "P"]:
            raise ValueError(
                f"Invalid long option type: {long_option_type}. Only 'C' (call) and 'P' (put) are supported."
            )
        if short_option_type not in ["C", "P"]:
            raise ValueError(
                f"Invalid short option type: {short_option_type}. Only 'C' (call) and 'P' (put) are supported."
            )

        if long_option_type != short_option_type:
            raise ValueError(
                "Both options in the spread must be of the same type (either both calls or both puts)."
            )

        if long_option_type == "P":
            if long_strike_price < short_strike_price:
                spread_type = "bullish"
            else:
                spread_type = "bearish"
        else:
            if long_strike_price > short_strike_price:
                spread_type = "bullish"
            else:
                spread_type = "bearish"

        if is_entry:
            if spread_type == "bullish" and long_option_type == "P":
                order_type = "NET_CREDIT"
                long_instruction = "BUY_TO_OPEN"
                short_instruction = "SELL_TO_OPEN"
            elif spread_type == "bearish" and long_option_type == "C":
                order_type = "NET_CREDIT"
                long_instruction = "BUY_TO_OPEN"
                short_instruction = "SELL_TO_OPEN"
            else:
                order_type = "NET_DEBIT"
                long_instruction = "BUY_TO_OPEN"
                short_instruction = "SELL_TO_OPEN"
        else:
            if spread_type == "bullish" and long_option_type == "P":
                order_type = "NET_DEBIT"
                long_instruction = "SELL_TO_CLOSE"
                short_instruction = "BUY_TO_CLOSE"
            elif spread_type == "bearish" and long_option_type == "C":
                order_type = "NET_DEBIT"
                long_instruction = "SELL_TO_CLOSE"
                short_instruction = "BUY_TO_CLOSE"
            else:
                order_type = "NET_CREDIT"
                long_instruction = "SELL_TO_CLOSE"
                short_instruction = "BUY_TO_CLOSE"

        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.strftime("%y%m%d")
        elif isinstance(expiration, str):
            if len(expiration) != 6:
                expiration = pd.Timestamp(expiration).strftime("%y%m%d")

        long_option_symbol = f"{symbol.ljust(6)}{expiration}{long_option_type}{str(int(long_strike_price * 1000)).zfill(8)}"
        short_option_symbol = f"{symbol.ljust(6)}{expiration}{short_option_type}{str(int(short_strike_price * 1000)).zfill(8)}"
        payload = {
            "orderType": order_type,
            "session": "NORMAL",
            "price": f"{price:.2f}",
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": long_instruction,
                    "quantity": quantity,
                    "instrument": {"symbol": long_option_symbol, "assetType": "OPTION"},
                },
                {
                    "instruction": short_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": short_option_symbol,
                        "assetType": "OPTION",
                    },
                },
            ],
        }
        return payload

    @classmethod
    def generate_iron_condor_json(
        cls,
        symbol,
        expiration,
        long_call_strike_price,
        short_call_strike_price,
        short_put_strike_price,
        long_put_strike_price,
        quantity,
        price,
        duration,
        is_entry=True,
    ):
        """
        Generate JSON for an iron condor trade.

        Args:
            symbol (str): The underlying symbol.
            expiration (str): The expiration date in YYMMDD format.
            long_call_strike_price (float): The strike price of the long call option.
            short_call_strike_price (float): The strike price of the short call option.
            short_put_strike_price (float): The strike price of the short put option.
            long_put_strike_price (float): The strike price of the long put option.
            quantity (int): The number of contracts.
            price (float): The price for the spread.
            duration (str): The duration of the order (e.g., 'DAY', 'GOOD_TILL_CANCEL').

        Returns:
            dict: The JSON payload for the trade.
        """
        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.strftime("%y%m%d")
        elif isinstance(expiration, str):
            if len(expiration) != 6:
                expiration = pd.Timestamp(expiration).strftime("%y%m%d")

        long_call_option_symbol = f"{symbol.ljust(6)}{expiration}C{str(int(long_call_strike_price * 1000)).zfill(8)}"
        short_call_option_symbol = f"{symbol.ljust(6)}{expiration}C{str(int(short_call_strike_price * 1000)).zfill(8)}"
        short_put_option_symbol = f"{symbol.ljust(6)}{expiration}P{str(int(short_put_strike_price * 1000)).zfill(8)}"
        long_put_option_symbol = f"{symbol.ljust(6)}{expiration}P{str(int(long_put_strike_price * 1000)).zfill(8)}"

        if is_entry:
            order_type = "NET_CREDIT"
            long_call_instruction = "BUY_TO_OPEN"
            short_call_instruction = "SELL_TO_OPEN"
            short_put_instruction = "SELL_TO_OPEN"
            long_put_instruction = "BUY_TO_OPEN"
        else:
            order_type = "NET_DEBIT"
            long_call_instruction = "SELL_TO_CLOSE"
            short_call_instruction = "BUY_TO_CLOSE"
            short_put_instruction = "BUY_TO_CLOSE"
            long_put_instruction = "SELL_TO_CLOSE"

        payload = {
            "orderType": order_type,
            "session": "NORMAL",
            "price": f"{price:.2f}",
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": long_call_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": long_call_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": short_call_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": short_call_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": short_put_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": short_put_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": long_put_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": long_put_option_symbol,
                        "assetType": "OPTION",
                    },
                },
            ],
        }
        return payload

    @classmethod
    def generate_straddle_json(
        cls,
        symbol,
        expiration,
        strike_price,
        quantity,
        price,
        duration,
        is_entry=True,
    ):
        """
        Generate JSON for a straddle trade.

        Args:
            symbol (str): The underlying symbol.
            expiration (str): The expiration date in YYMMDD format.
            strike_price (float): The strike price of the call and put options.
            quantity (int): The number of contracts.
            price (float): The price for the spread.
            duration (str): The duration of the order (e.g., 'DAY', 'GOOD_TILL_CANCEL').

        Returns:
            dict: The JSON payload for the trade.
        """
        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.strftime("%y%m%d")
        elif isinstance(expiration, str):
            if len(expiration) != 6:
                expiration = pd.Timestamp(expiration).strftime("%y%m%d")

        call_option_symbol = f"{symbol.ljust(6)}{expiration}C{str(int(strike_price * 1000)).zfill(8)}"
        put_option_symbol = f"{symbol.ljust(6)}{expiration}P{str(int(strike_price * 1000)).zfill(8)}"

        if is_entry:
            order_type = "NET_DEBIT"
            call_instruction = "BUY_TO_OPEN"
            put_instruction = "BUY_TO_OPEN"
        else:
            order_type = "NET_CREDIT"
            call_instruction = "SELL_TO_CLOSE"
            put_instruction = "SELL_TO_CLOSE"

        payload = {
            "orderType": order_type,
            "session": "NORMAL",
            "price": f"{price:.2f}",
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": call_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": call_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": put_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": put_option_symbol,
                        "assetType": "OPTION",
                    },
                },
            ],
        }
        return payload

    @classmethod
    def generate_iron_butterfly_json(
        cls,
        symbol,
        expiration,
        long_call_strike_price,
        strike_price,
        long_put_strike_price,
        quantity,
        price,
        duration,
        is_entry=True,
    ):
        """
        Generate JSON for an iron butterfly spread trade.

        Args:
            symbol (str): The underlying symbol.
            expiration (str): The expiration date in YYMMDD format.
            long_call_strike_price (float): The strike price of the long call options.
            strike_price (float): The strike price of the short call and short put options.
            long_put_strike_price (float): The strike price of the long put options.
            quantity (int): The number of contracts.
            price (float): The price for the spread.
            duration (str): The duration of the order (e.g., 'DAY', 'GOOD_TILL_CANCEL').

        Returns:
            dict: The JSON payload for the trade.
        """
        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.strftime("%y%m%d")
        elif isinstance(expiration, str):
            if len(expiration) != 6:
                expiration = pd.Timestamp(expiration).strftime("%y%m%d")

        long_call_option_symbol = f"{symbol.ljust(6)}{expiration}C{str(int(long_call_strike_price * 1000)).zfill(8)}"
        short_call_option_symbol = f"{symbol.ljust(6)}{expiration}C{str(int(strike_price * 1000)).zfill(8)}"
        long_put_option_symbol = f"{symbol.ljust(6)}{expiration}P{str(int(long_put_strike_price * 1000)).zfill(8)}"
        short_put_option_symbol = f"{symbol.ljust(6)}{expiration}P{str(int(strike_price * 1000)).zfill(8)}"

        if is_entry:
            order_type = "NET_CREDIT"
            long_call_instruction = "BUY_TO_OPEN"
            short_call_instruction = "SELL_TO_OPEN"
            long_put_instruction = "BUY_TO_OPEN"
            short_put_instruction = "SELL_TO_OPEN"
        else:
            order_type = "NET_DEBIT"
            long_call_instruction = "SELL_TO_CLOSE"
            short_call_instruction = "BUY_TO_CLOSE"
            long_put_instruction = "SELL_TO_CLOSE"
            short_put_instruction = "BUY_TO_CLOSE"

        payload = {
            "orderType": order_type,
            "session": "NORMAL",
            "price": f"{price:.2f}",
            "duration": duration,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": long_call_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": long_call_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": short_call_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": short_call_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": long_put_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": long_put_option_symbol,
                        "assetType": "OPTION",
                    },
                },
                {
                    "instruction": short_put_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": short_put_option_symbol,
                        "assetType": "OPTION",
                    },
                },
            ],
        }
        return payload

    def place_order(self, account_number_hash_value, payload):
        url = f"{self.trading_base_url}/accounts/{account_number_hash_value}/orders"
        response = self._post(url, data=payload)
        if response.status_code == 201:
            placed_time = pd.Timestamp(response.headers.get("Date")).tz_convert(
                "America/New_York"
            )  # Assuming the response contains the order url and timestamp
            order_link = response.headers.get("location")
            logger.info(
                f"Order placed successfully at {placed_time}. Link: {order_link}"
            )
            return order_link, placed_time
        else:
            response.raise_for_status()
            error_message = response.json().get("message", "Unknown error")
            error_details = response.json().get("errors", [])
            correl_id = response.headers.get("Schwab-Client-CorrelID", "Unknown")
            logger.warning(
                f"Order placement failed with status code {response.status_code}. "
                f"Error message: {error_message}. "
                f"Error details: {error_details}. "
                f"Correlation ID: {correl_id}"
            )
            return False

    def cancel_order(
        self, order_url=None, account_number_hash_value=None, order_id=None
    ):
        if order_url:
            if account_number_hash_value or order_id:
                raise ValueError(
                    "Provide either order_url OR (account_number_hash_value and order_id), not both."
                )
            url = order_url
        elif account_number_hash_value and order_id:
            url = f"{self.trading_base_url}/accounts/{account_number_hash_value}/orders/{order_id}"
        else:
            raise ValueError(
                "Provide either order_url OR both account_number_hash_value and order_id."
            )

        response = self._delete(url)
        if response.status_code == 200:
            placed_time = pd.Timestamp(response.headers.get("Date")).tz_convert(
                "America/New_York"
            )  # Assuming the response contains the timestamp
            logger.info(f"Order canceled successfully at {placed_time}.")
            return placed_time
        else:
            error_message = response.json().get("message", "Unknown error")
            error_details = response.json().get("errors", [])
            correl_id = response.headers.get("Schwab-Client-CorrelID", "Unknown")
            logger.warning(
                f"Order placement failed with status code {response.status_code}. "
                f"Error message: {error_message}. "
                f"Error details: {error_details}. "
                f"Correlation ID: {correl_id}"
            )
            return False

    def get_order(self, order_url=None, account_number_hash_value=None, order_id=None):
        if order_url:
            if account_number_hash_value or order_id:
                raise ValueError(
                    "Provide either order_url OR (account_number_hash_value and order_id), not both."
                )
            url = order_url
        elif account_number_hash_value and order_id:
            url = f"{self.trading_base_url}/accounts/{account_number_hash_value}/orders/{order_id}"
        else:
            raise ValueError(
                "Provide either order_url OR both account_number_hash_value and order_id."
            )

        response = self._get(url)
        if response.status_code == 200:
            placed_time = pd.Timestamp(response.headers.get("Date")).tz_convert(
                "America/New_York"
            )  # Assuming the response contains the timestamp
            logger.info(f"Order retrieved successfully at {placed_time}.")
            return response.json()
        else:
            error_message = response.json().get("message", "Unknown error")
            error_details = response.json().get("errors", [])
            correl_id = response.headers.get("Schwab-Client-CorrelID", "Unknown")
            logger.warning(
                f"Order placement failed with status code {response.status_code}. "
                f"Error message: {error_message}. "
                f"Error details: {error_details}. "
                f"Correlation ID: {correl_id}"
            )
            return False

    def modify_order(
        self, payload, order_url=None, account_number_hash_value=None, order_id=None
    ):
        if order_url:
            if account_number_hash_value or order_id:
                raise ValueError(
                    "Provide either order_url OR (account_number_hash_value and order_id), not both."
                )
            url = order_url
        elif account_number_hash_value and order_id:
            url = f"{self.trading_base_url}/accounts/{account_number_hash_value}/orders/{order_id}"
        else:
            raise ValueError(
                "Provide either order_url OR both account_number_hash_value and order_id."
            )
        response = self._put(url, data=payload)
        if response.status_code == 201:
            placed_time = pd.Timestamp(response.headers.get("Date")).tz_convert(
                "America/New_York"
            )  # Assuming the response contains the order url and timestamp
            order_link = response.headers.get("location")
            logger.info(
                f"Order placed successfully at {placed_time}. Link: {order_link}"
            )
            return order_link, placed_time
        else:
            error_message = response.json().get("message", "Unknown error")
            error_details = response.json().get("errors", [])
            correl_id = response.headers.get("Schwab-Client-CorrelID", "Unknown")
            logger.warning(
                f"Order placement failed with status code {response.status_code}. "
                f"Error message: {error_message}. "
                f"Error details: {error_details}. "
                f"Correlation ID: {correl_id}"
            )
            return False


def main():

    client_id = os.environ.get("SCHWAB_CLIENT_ID")
    client_secret = os.environ.get("SCHWAB_CLIENT_SECRET")
    schwab_trade = SchwabTrade(client_id, client_secret, token_file="token.json")

    # Test 1: Get account numbers
    account_numbers = schwab_trade.get_account_numbers()
    assert len(account_numbers) == 2, "Expected 2 account numbers"
    assert (
        "accountNumber" in account_numbers[0]
    ), "Expected 'accountNumber' key in account numbers 1"
    assert (
        "hashValue" in account_numbers[0]
    ), "Expected 'hashValue' key in account numbers 1"
    assert (
        "accountNumber" in account_numbers[1]
    ), "Expected 'accountNumber' key in account numbers 2"
    assert (
        "hashValue" in account_numbers[1]
    ), "Expected 'hashValue' key in account numbers 2"

    # Test 2: Refresh token method
    assert schwab_trade.refresh_token(), "Expected refresh token to return True"

    # Test 3: Get Account Info
    account = schwab_trade.get_specific_account_info(
        schwab_trade.account_numbers[0]["hashValue"]
    )
    schwab_trade.account_numbers[0]["hashValue"]
    assert account, "Some returned value"
    assert type(account) is dict, "Expected dictionary type"

    # Test 4: Get all account info
    all_accounts = schwab_trade.get_all_account_info()
    assert all_accounts, "Some returned value"
    assert type(all_accounts) is list, "Expected dictionary type"

    # Test 5: Get orders
    orders = schwab_trade.get_orders(account_numbers[0]["hashValue"])
    assert orders, "Some returned value"

    # Test 6: Place single-option order
    payload1 = schwab_trade.generate_single_option_json(
        "SPY", "241031", "C", 562, "BUY_TO_OPEN", 1, "LIMIT", 1, "GOOD_TILL_CANCEL"
    )
    payload2 = schwab_trade.generate_single_option_json(
        "SPY", "2024-10-31", "C", 562, "BUY_TO_OPEN", 1, "LIMIT", 1, "GOOD_TILL_CANCEL"
    )
    payload3 = schwab_trade.generate_single_option_json(
        "SPY",
        pd.Timestamp("2024-10-31"),
        "C",
        562,
        "BUY_TO_OPEN",
        1,
        "LIMIT",
        1,
        "GOOD_TILL_CANCEL",
    )
    assert payload1 == payload2 == payload3, "Expected all payloads to be equal"
    order_link = schwab_trade.place_order(account_numbers[0]["hashValue"], payload1)[0]
    assert order_link != "", "Expected non-empty response"

    # Test 7: Get order
    order = schwab_trade.get_order(order_link)

    # Test 8: Cancel order
    assert schwab_trade.cancel_order(order_link) != "", "Expected empty response"
    order = schwab_trade.get_order(order_link)
    assert order.get("status") == "CANCELED", "Expected order to be canceled"

    # Test 9: Place Vertical spreads
    payload1 = schwab_trade.generate_vertical_spread_json(
        "SPY", "241115", "P", 561, "P", 562, 1, 0.22, "GOOD_TILL_CANCEL", is_entry=True
    )
    payload2 = schwab_trade.generate_vertical_spread_json(
        "SPY",
        "20241115",
        "P",
        561,
        "P",
        562,
        1,
        0.22,
        "GOOD_TILL_CANCEL",
        is_entry=True,
    )
    payload3 = schwab_trade.generate_vertical_spread_json(
        "SPY",
        "2024-11-15",
        "P",
        561,
        "P",
        562,
        1,
        0.22,
        "GOOD_TILL_CANCEL",
        is_entry=True,
    )
    payload4 = schwab_trade.generate_vertical_spread_json(
        "SPY",
        pd.Timestamp("2024-11-15"),
        "P",
        561,
        "P",
        562,
        1,
        0.22,
        "GOOD_TILL_CANCEL",
        is_entry=True,
    )
    assert payload1 == payload2 == payload3 == payload4, "Expected some payload"
    order_link = schwab_trade.place_order(account_numbers[0]["hashValue"], payload1)[0]
    assert order_link != "", "Expected non-empty response"
    order = schwab_trade.get_order(order_link)
    assert order.get("status") in [
        "SUBMITTED",
        "PENDING_ACTIVATION",
        "WORKING",
        "FILLED",
    ], "Expected order to be submitted"

    # Test 10: close order
    closing_payload = schwab_trade.generate_vertical_spread_json(
        "SPY", "241115", "P", 561, "P", 562, 1, 0.24, "GOOD_TILL_CANCEL", is_entry=False
    )
    order_link = schwab_trade.place_order(
        account_numbers[0]["hashValue"], closing_payload
    )[0]
    assert order_link != "", "Expected non-empty response"
    order = schwab_trade.get_order(order_link)
    assert order.get("status") in [
        "SUBMITTED",
        "PENDING_ACTIVATION",
        "WORKING",
        "FILLED",
    ], "Expected order to be submitted"
    # schwab_trade.cancel_order(order_link)

    # Test 10: Cancel vertical spread order
    assert schwab_trade.cancel_order(order_link) != "", "Expected empty response"
    order = schwab_trade.get_order(order_link)
    assert order.get("status") == "CANCELED", "Expected order to be canceled"

    # Test 11: Modify a vertical spread order
    payload = schwab_trade.generate_vertical_spread_json(
        "SPY",
        pd.Timestamp("2024-10-31"),
        "P",
        560,
        "P",
        562,
        1,
        "NET_CREDIT",
        1,
        "GOOD_TILL_CANCEL",
    )
    order_link = schwab_trade.place_order(account_numbers[0]["hashValue"], payload)[0]
    assert order_link != "", "Expected non-empty response"
    order = schwab_trade.get_order(order_link)
    assert order.get("status") in [
        "SUBMITTED",
        "PENDING_ACTIVATION",
        "WORKING",
        "PENDING_ACKNOWLEDGEMENT",
    ], "Expected order to be submitted"

    ## Modifying part: credit
    replacement_payload = schwab_trade.generate_vertical_spread_json(
        "SPY",
        pd.Timestamp("2024-10-31"),
        "P",
        560,
        "P",
        562,
        1,
        "NET_CREDIT",
        1.5,
        "GOOD_TILL_CANCEL",
    )
    replacement_order_link = schwab_trade.modify_order(replacement_payload, order_link)[
        0
    ]
    assert replacement_order_link != "", "Expected non-empty response"
    replacement_order = schwab_trade.get_order(replacement_order_link)

    assert replacement_order.get("status") in [
        "SUBMITTED",
        "PENDING_ACTIVATION",
        "WORKING",
        "PENDING_ACKNOWLEDGEMENT",
    ], "Expected order to be submitted"

    ## Modifying another time: strikes
    replacement_payload2 = schwab_trade.generate_vertical_spread_json(
        "SPY",
        pd.Timestamp("2024-10-31"),
        "P",
        559,
        "P",
        561,
        1,
        "NET_CREDIT",
        1.5,
        "GOOD_TILL_CANCEL",
    )
    replacement_order_link2 = schwab_trade.modify_order(
        replacement_payload2, replacement_order_link
    )[0]
    assert replacement_order_link2 != "", "Expected non-empty response"
    replacement_order2 = schwab_trade.get_order(replacement_order_link2)
    assert replacement_order2.get("status") in [
        "SUBMITTED",
        "PENDING_ACTIVATION",
        "WORKING",
    ], "Expected order to be submitted"

    assert (
        schwab_trade.cancel_order(replacement_order_link2) != ""
    ), "Expected empty response"
    replacement_order2 = schwab_trade.get_order(replacement_order_link2)
    assert (
        replacement_order2.get("status") == "CANCELED"
    ), "Expected order to be canceled"

    print("All tests passed!")


if __name__ == "__main__":
    main()
