import pandas as pd
from .option_manager import OptionBacktester, Config
from ..brokers.schwab.schwab_order import SchwabOptionOrder
from ..brokers.schwab.schwab_auth import SchwabAuth
from ..brokers.schwab.schwab_data import SchwabData
from ..brokers.broker import OptionBroker
from ..brokers.order import Order
from typing import List
import dill
from loguru import logger


class TradingManager(OptionBacktester):

    def __init__(self, config: Config):
        super().__init__(config)
        self.active_orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.option_broker = OptionBroker(config)
        self.__dict__.update(config.__dict__)

    def add_order(self, order: Order) -> bool:
        """Add an order to the list of active orders."""
        market_isopen = order.market_isOpen()
        if market_isopen:
            if self.add_spread(order):
                order.submit_entry()
                self.active_orders.append(order)
                return True
        return False

    def update_orders(self, current_time, option_chain_df=None):
        """Update the status of all orders."""

        if self.active_orders:
            if self.active_orders[0].market_isOpen():
                if option_chain_df is None:
                    option_chain_df = self.option_broker.data.get_option_chain(
                        self.ticker
                    )
                for order in self.active_orders:
                    orders_to_close = []
                    order.update_order(option_chain_df)  # update the order part
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

                self.update(current_time, option_chain_df)  # update the strategy part

    def get_active_orders(self) -> List[Order]:
        return self.active_orders

    def get_closed_orders(self) -> List[Order]:
        return self.closed_orders

    def freeze(self, file_path: str) -> None:
        """Save the TradingManager instance to a dill file."""
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    def get_orders_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame containing important information about the active orders."""
        columns = [
            "Order ID",
            "Symbol",
            "Strategy Type",
            "Description",
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
            "Return (%)",
            "Total Commission",
            "DIT",
        ]
        data = []

        for order in self.active_orders + self.closed_orders:
            data.append(
                [
                    order.order_id.split("/")[
                        -1
                    ],  # Using id to uniquely identify the order
                    order.symbol,
                    order.strategy_type,
                    order.short_description(),
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
                    order.return_percentage(),
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
            "Description",
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
            "Return (%)",
            "Total Commission",
            "DIT",
        ]
        data = []

        for order in self.active_orders:
            data.append(
                [
                    order.order_id.split("/")[
                        -1
                    ],  # Using id to uniquely identify the order
                    order.symbol,
                    order.strategy_type,
                    order.short_description(),
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
                    order.return_percentage(),
                    order.calculate_total_commission(),
                    order.DIT,
                ]
            )

        return pd.DataFrame(data, columns=columns)

    def auth_refresh(self):
        """Refresh the authentication for all active and closed orders."""
        # TODO: Add abstract class for auth. Make sure authentication and refreshing works
        try:
            self.option_broker.auth.refresh_access_token()
        except Exception as e:
            logger.exception(
                "... Attempting to get new token by going through the authentication process"
            )
            self.option_broker.auth.authenticate()

        for order in self.active_orders:
            order.auth = self.option_broker.auth
        for order in self.closed_orders:
            order.auth = self.option_broker.auth

        self.option_broker.data.auth = self.option_broker.auth
        self.option_broker.trading.auth = self.option_broker.auth

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by its ID."""
        order_to_cancel = None
        for i, order in enumerate(self.active_orders):
            if order.order_id.split("/")[-1] == order_id:
                order_to_cancel = order
                break

        if order_to_cancel:
            order_to_cancel.cancel()
            self.active_orders.pop(i)
            return True
        else:
            logger.warning(f"Order with ID {order_id} not found.")
            return False

    def close_order(self, order_id: str) -> bool:
        """Close an order by its ID."""
        order_to_close = None
        for order in self.active_orders:
            if order.order_id.split("/")[-1] == order_id:
                order_to_close = order
                break

        if order_to_close:
            order_to_close.close_order()
            self.closed_orders.append(order_to_close)
            self.active_orders = [
                order for order in self.active_orders if order.status != "CLOSED"
            ]
            return True
        else:
            logger.warning(f"Order with ID {order_id} not found.")
            return False

    def liquidate_all(self) -> None:
        """Close all active orders."""
        for order in self.active_orders[:]:  # Iterate over a copy to avoid modifying the list while iterating
            self.close_order(order.order_id.split("/")[-1])

    def update_config(self, config: Config) -> None:
        """
        Update the trading manager's configuration.
        
        Args:
            config (Config): New configuration object
        """
        self.config = config
        self.__dict__.update(config.__dict__)
        self.option_broker.update_config(config)
