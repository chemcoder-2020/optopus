import pandas as pd
from option_manager import OptionBacktester
from schwab_order import SchwabOptionOrder
from schwab_auth import SchwabAuth
from schwab_data import SchwabData
from typing import List
import pickle
import os


class TradingManager(OptionBacktester):
    def __init__(self, config):
        super().__init__(config)
        self.active_orders: List[SchwabOptionOrder] = []
        self.closed_orders: List[SchwabOptionOrder] = []
        self.schwab_data = SchwabData(
            client_id=os.getenv("SCHWAB_CLIENT_ID"),
            client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
            redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
            token_file="token.json",
        )

    def add_order(self, order: SchwabOptionOrder) -> bool:
        if order.market_isOpen():
            if self.add_spread(order):
                order.submit_entry()
                self.active_orders.append(order)
                return True
        return False

    def update_orders(self, current_time, option_chain_df=None):
        if self.active_orders:
            if self.active_orders[0].market_isOpen():
                for order in self.active_orders:
                    orders_to_close = []
                    order.update_order(option_chain_df)
                    if order.status == "CLOSED":
                        order.submit_exit()
                        orders_to_close.append(order)

                    if orders_to_close:
                        self.closed_orders.extend(orders_to_close)
                        self.active_orders = [
                            order
                            for order in self.active_orders
                            if order.status != "CLOSED"
                        ]

    def get_active_orders(self) -> List[SchwabOptionOrder]:
        return self.active_orders

    def get_closed_orders(self) -> List[SchwabOptionOrder]:
        return self.closed_orders

    def freeze(self, file_path: str) -> None:
        """Save the TradingManager instance to a pickle file."""
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def get_orders_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame containing important information about the active orders."""
        columns = [
            "Order ID",
            "Symbol",
            "Strategy Type",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Current Time",
            "Exit Time",
            "Order Status",
            "Status",
            "Profit Target",
            "Bid",
            "Ask",
            "Price",
            "Total P/L",
            "Total Commission",
            "DIT",
        ]
        data = []

        for order in self.active_orders+self.closed_orders:
            data.append(
                [
                    order.order_id.split("/")[-1],  # Using id to uniquely identify the order
                    order.symbol,
                    order.strategy_type,
                    order.contracts,
                    order.entry_time,
                    order.entry_net_premium,
                    order.current_time,
                    order.exit_time if hasattr(order, "exit_time") else None,
                    order.order_status,
                    order.status,
                    order.profit_target,
                    order.current_bid,
                    order.current_ask,
                    order.net_premium,
                    order.total_pl(),
                    order.calculate_total_commission(),
                    order.DIT,
                ]
            )

        return pd.DataFrame(data, columns=columns)
    
    def get_active_orders_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame containing important information about the active orders."""
        columns = [
            "Order ID",
            "Symbol",
            "Strategy Type",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Current Time",
            "Exit Time",
            "Order Status",
            "Status",
            "Profit Target",
            "Bid",
            "Ask",
            "Price",
            "Total P/L",
            "Total Commission",
            "DIT",
        ]
        data = []

        for order in self.active_orders:
            data.append(
                [
                    order.order_id.split("/")[-1],  # Using id to uniquely identify the order
                    order.symbol,
                    order.strategy_type,
                    order.contracts,
                    order.entry_time,
                    order.entry_net_premium,
                    order.current_time,
                    order.exit_time if hasattr(order, "exit_time") else None,
                    order.order_status,
                    order.status,
                    order.profit_target,
                    order.current_bid,
                    order.current_ask,
                    order.net_premium,
                    order.total_pl(),
                    order.calculate_total_commission(),
                    order.DIT,
                ]
            )

        return pd.DataFrame(data, columns=columns)

    def auth_refresh(
        self, client_id, client_secret, redirect_uri, token_file="token.json"
    ):
        """Refresh the authentication for all active and closed orders."""
        new_auth = SchwabAuth(client_id, client_secret, redirect_uri, token_file)
        try:
            new_auth.refresh_access_token()
        except Exception as e:
            self.logger.info(
                f"{e}... Attempting to get new token by going through the authentication process"
            )
            new_auth.authenticate()
        for order in self.active_orders:
            order.auth = new_auth
        for order in self.closed_orders:
            order.auth = new_auth

        self.schwab_data.auth = new_auth
