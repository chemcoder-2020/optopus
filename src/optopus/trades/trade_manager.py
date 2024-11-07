import pandas as pd
from .option_manager import OptionBacktester, Config
from .option_leg import calculate_dte
from ..brokers.schwab.schwab_order import SchwabOptionOrder
from ..brokers.schwab.schwab_auth import SchwabAuth
from ..brokers.schwab.schwab_data import SchwabData
from ..brokers.broker import OptionBroker
from ..brokers.order import Order
from typing import List
import dill
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


class TradingManager(OptionBacktester):

    def __init__(self, config: Config, name: str = "TradingManager"):
        super().__init__(config)
        self.active_orders: List[Order] = self.active_trades
        self.closed_orders: List[Order] = self.closed_trades
        self.option_broker = OptionBroker(config)
        self.__dict__.update(config.__dict__)
        self.automation_on = True
        self.management_on = True
        self.name = name

    def add_order(self, order: Order) -> bool:
        """Add an order to the list of active orders."""
        market_isopen = order.market_isOpen()
        if market_isopen and (
            hasattr(self, "automation_on") == False or self.automation_on
        ):
            if self.add_spread(order):
                if order.submit_entry():
                    # self.active_orders.append(order)
                    return True
                else:
                    self.active_trades.pop()  # if not submitted, remove from active trades (which would remove from active orders because they're identical)
                    self.available_to_trade += (
                        order.get_required_capital()
                    )  # recover available capital
                    self._update_trade_counts()
                    logger.info(
                        f"Entry failed. Removed from active trades and recovered capital."
                    )
                    return False
        return False

    def _process_order(self, order, option_chain_df=None):
        """Helper function to process individual orders."""
        order.update_order(option_chain_df)
        if order.status == "CLOSED":
            if order.exit_order_id:
                logger.info(f"Closed order {order.order_id}.")
                return order
        logger.info(f"Successfully updated order {order.order_id}.")
        return None

    def update_orders(self, option_chain_df=None):
        """Update the status of all orders using parallel processing."""
        if (
            self.active_orders
            and self.active_orders[0].market_isOpen()
            and (hasattr(self, "management_on") == False or self.management_on)
        ):

            with ThreadPoolExecutor() as executor:
                orders_to_close = list(
                    executor.map(
                        lambda order: self._process_order(order, option_chain_df),
                        self.active_orders,
                    )
                )

            orders_to_close = [order for order in orders_to_close if order]
            if orders_to_close:
                self.closed_orders.extend(orders_to_close)
                self.active_orders = [
                    order for order in self.active_orders if order.status != "CLOSED"
                ]

            self._update_trade_counts()

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
            "Expiration",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Current Time",
            "Exit Time",
            "Exit Order ID",
            "Order Status",
            "Status",
            "Profit Target",
            "Bid",
            "Ask",
            "Price",
            "Total P/L",
            "Return (%)",
            "Highest Return",
            "Total Commission",
            "DIT",
            "DTE",
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
                    order.min_expiration,
                    order.contracts,
                    order.entry_time,
                    order.entry_net_premium,
                    order.current_time,
                    order.exit_time if hasattr(order, "exit_time") else None,
                    (
                        order.exit_order_id.split("/")[-1]
                        if hasattr(order, "exit_order_id") and order.exit_order_id
                        else None
                    ),
                    order.order_status,
                    order.status,
                    (
                        order.exit_scheme.profit_target
                        if hasattr(order, "exit_scheme")
                        and hasattr(order.exit_scheme, "profit_target")
                        else order.profit_target
                    ),
                    order.current_bid,
                    order.current_ask,
                    order.net_premium,
                    order.total_pl(),
                    round(order.return_percentage(), 1),
                    round(order.highest_return, 1),
                    order.calculate_total_commission(),
                    order.DIT,
                    round(calculate_dte(order.min_expiration, order.current_time)),
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
            "Expiration",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Current Time",
            "Exit Time",
            "Exit Order ID",
            "Order Status",
            "Status",
            "Profit Target",
            "Bid",
            "Ask",
            "Price",
            "Total P/L",
            "Return (%)",
            "Highest Return",
            "Total Commission",
            "DIT",
            "DTE",
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
                    order.min_expiration,
                    order.contracts,
                    order.entry_time,
                    order.entry_net_premium,
                    order.current_time,
                    order.exit_time if hasattr(order, "exit_time") else None,
                    (
                        order.exit_order_id.split("/")[-1]
                        if hasattr(order, "exit_order_id") and order.exit_order_id
                        else None
                    ),
                    order.order_status,
                    order.status,
                    (
                        order.exit_scheme.profit_target
                        if hasattr(order, "exit_scheme")
                        and hasattr(order.exit_scheme, "profit_target")
                        else order.profit_target
                    ),
                    order.current_bid,
                    order.current_ask,
                    order.net_premium,
                    order.total_pl(),
                    round(order.return_percentage(), 1),
                    round(order.highest_return, 1),
                    order.calculate_total_commission(),
                    order.DIT,
                    round(calculate_dte(order.min_expiration, order.current_time)),
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
            self.active_orders.remove(order_to_close)
            self.available_to_trade += order_to_close.get_required_capital()
            return True
        else:
            logger.warning(f"Order with ID {order_id} not found.")
            return False

    def liquidate_all(self) -> None:
        """Close all active orders."""
        for order in self.active_orders[
            :
        ]:  # Iterate over a copy to avoid modifying the list while iterating
            self.close_order(order.order_id.split("/")[-1])

    def override_order(self, order_id: str) -> bool:
        """Override an order by its ID, removing it from all lists."""
        order_to_override = None
        for i, order in enumerate(self.active_orders):
            if order.order_id.split("/")[-1] == order_id:
                order_to_override = order
                break

        if order_to_override:
            self.active_orders.pop(i)
            self.available_to_trade += order_to_override.get_required_capital()
            return True

        for i, order in enumerate(self.closed_orders):
            if order.order_id.split("/")[-1] == order_id:
                order_to_override = order
                break

        if order_to_override:
            self.closed_orders.pop(i)
            self.available_to_trade += order_to_override.get_required_capital()
            return True

        self.active_trades = self.active_orders
        self.closed_trades = self.closed_orders  # sync active and closed trades

        logger.warning(f"Order with ID {order_id} not found.")
        return False
