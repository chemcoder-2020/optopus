import pandas as pd
from .option_manager import OptionBacktester, Config
from .option_leg import calculate_dte
from .strategies.vertical_spread import VerticalSpread
from ..brokers.broker import OptionBroker
from ..brokers.order import Order
from typing import List
import dill
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


class TradingManager(OptionBacktester):

    def __init__(self, config: Config, name: str = "TradingManager"):
        """
        Initialize the TradingManager class.

        Parameters:
        - config (Config): Configuration parameters.
        - name (str, optional): Name of the trading manager. Defaults to "TradingManager".
        """
        super().__init__(config)
        self.active_orders: List[Order] = self.active_trades
        self.closed_orders: List[Order] = self.closed_trades
        self.option_broker = OptionBroker(config)
        self.__dict__.update(config.__dict__)
        if not hasattr(config, "trade_type"):
            raise ValueError("Config must contain a trade_type attribute.")
        self.trade_type = config.trade_type
        self.automation_on = True
        self.management_on = True
        self.name = name

    def add_order(self, order: Order) -> bool:
        """
        Add an order to the list of active orders.

        Parameters:
        - order (Order): The order to add.

        Returns:
        - bool: True if the order was successfully added, False otherwise.
        """
        market_isopen = order.market_isOpen()
        logger.info(f"Market open: {market_isopen}")
        if market_isopen and (
            hasattr(self, "automation_on") is False or self.automation_on
        ):
            if self.add_spread(order):
                if order.submit_entry():
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
        """
        Helper function to process individual orders.

        Parameters:
        - order (Order): The order to process.
        - option_chain_df (pd.DataFrame, optional): The option chain DataFrame. Defaults to None.

        Returns:
        - Order or None: The closed order if it was closed, None otherwise.
        """
        order.update_order(option_chain_df)
        if order.status == "CLOSED":
            if order.exit_order_id:
                logger.info(f"Closed order {order.order_id}.")
                return order
        logger.info(f"Successfully updated order {order.order_id}.")
        return None

    def update_orders(self, option_chain_df=None):
        """
        Update the status of all orders using parallel processing.

        Parameters:
        - option_chain_df (pd.DataFrame, optional): The option chain DataFrame. Defaults to None.
        """
        if (
            self.active_orders
            and self.active_orders[0].market_isOpen()
            and (hasattr(self, "management_on") == False or self.management_on)
        ):
            current_time = self.active_orders[-1].current_time

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
                for order in orders_to_close:
                    self.active_orders.remove(order)

            self._update_trade_counts()

            self.last_update_time = current_time

            # Record performance data after update
            if isinstance(self.last_update_time, pd.Timestamp):
                self._record_performance_data(current_time)

    def _record_performance_data(self, current_time):
        """
        Record performance data for the given time.

        Parameters:
        - current_time (datetime): The current time.
        """
        total_pl = self.get_total_pl()
        closed_pl = self.get_closed_pl()
        active_positions = len(self.active_trades)
        self.performance_data.append(
            {
                "time": current_time,
                "total_pl": total_pl,
                "closed_pl": closed_pl,
                "active_positions": active_positions,
            }
        )

    def get_active_orders(self) -> List[Order]:
        """
        Get the list of active orders.

        Returns:
        - List[Order]: List of active orders.
        """
        return self.active_orders

    def get_closed_orders(self) -> List[Order]:
        """
        Get the list of closed orders.

        Returns:
        - List[Order]: List of closed orders.
        """
        return self.closed_orders

    def freeze(self, file_path: str) -> None:
        """
        Save the TradingManager instance to a dill file.

        Parameters:
        - file_path (str): Path to the file where the instance will be saved.
        """
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    def get_orders_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing important information about the active and closed orders.

        Returns:
        - pd.DataFrame: DataFrame with order information.
        """
        columns = [
            "Order ID",
            "Symbol",
            "Strategy Type",
            "Description",
            "Expiration",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Entry Bid",
            "Entry Ask",
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
            "Entry Delta",
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
                    order.entry_bid,
                    order.entry_ask,
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
                    order.entry_delta if hasattr(order, "entry_delta") else None,
                ]
            )

        return pd.DataFrame(data, columns=columns)

    def get_active_orders_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing important information about the active orders.

        Returns:
        - pd.DataFrame: DataFrame with order information.
        """
        columns = [
            "Order ID",
            "Symbol",
            "Strategy Type",
            "Description",
            "Expiration",
            "Contracts",
            "Entry Time",
            "Entry Price",
            "Entry Bid",
            "Entry Ask",
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
            "Entry Delta",
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
                    order.entry_bid,
                    order.entry_ask,
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
                    order.entry_delta if hasattr(order, "entry_delta") else None,
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

    def authenticate(self):
        """Refresh the authentication for all active and closed orders."""

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
            self._update_trade_counts()
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
            self._update_trade_counts()
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
        self._update_trade_counts()

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
            self._update_trade_counts()
            return True

        for i, order in enumerate(self.closed_orders):
            if order.order_id.split("/")[-1] == order_id:
                order_to_override = order
                break

        if order_to_override:
            self.closed_orders.pop(i)
            self.available_to_trade += order_to_override.get_required_capital()
            self._update_trade_counts()
            return True

        # self.active_trades = self.active_orders
        # self.closed_trades = self.closed_orders  # sync active and closed trades

        logger.warning(f"Order with ID {order_id} not found.")
        return False

    def next(self, STRATEGY_PARAMS: dict) -> None:
        """Update orders and check for new entries."""
        trade_type_required_keys = {
            "Vertical Spread": {
                "option_type",
                "long_delta",
                "short_delta",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
            },
            "Naked Put": {
                "strike",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
                "strategy_side",
            },
            "Naked Call": {
                "strike",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
                "strategy_side",
            },
            "Iron Condor": {
                "put_long_strike",
                "put_short_strike",
                "call_short_strike",
                "call_long_strike",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
            },
            "Iron Butterfly": {
                "lower_strike",
                "middle_strike",
                "upper_strike",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
                "strategy_side",
            },
            "Straddle": {
                "strike",
                "dte",
                "contracts",
                "commission",
                "exit_scheme",
                "strategy_side",
            },
        }

        if self.trade_type not in trade_type_required_keys:
            logger.error(f"Unsupported trade type: {self.trade_type}")
            return

        required_keys = trade_type_required_keys[self.trade_type]
        if not required_keys.issubset(STRATEGY_PARAMS):
            logger.error(
                f"Missing keys in STRATEGY_PARAMS for {self.trade_type}: {required_keys - STRATEGY_PARAMS}"
            )
            return

        if self.management_on:
            self.update_orders()

        if self.automation_on:
            option_chain_df = self.option_broker.data.get_option_chain(
                self.config.ticker,
                strike_count=STRATEGY_PARAMS.get("chain_strike_count", 50),
            )
            bar = option_chain_df["QUOTE_READTIME"].iloc[0]

            if self.trade_type == "Vertical Spread":
                strategy = VerticalSpread.create_vertical_spread(
                    symbol=self.config.ticker,
                    option_type=STRATEGY_PARAMS.get("option_type"),
                    long_strike=STRATEGY_PARAMS.get("long_delta"),
                    short_strike=STRATEGY_PARAMS.get("short_delta"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                )
            elif self.trade_type == "Naked Put":
                from .strategies.naked_put import NakedPut

                strategy = NakedPut.create_naked_put(
                    symbol=self.config.ticker,
                    strike=STRATEGY_PARAMS.get("strike"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                    strategy_side=STRATEGY_PARAMS.get("strategy_side"),
                )
            elif self.trade_type == "Naked Call":
                from .strategies.naked_call import NakedCall

                strategy = NakedCall.create_naked_call(
                    symbol=self.config.ticker,
                    strike=STRATEGY_PARAMS.get("strike"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                    strategy_side=STRATEGY_PARAMS.get("strategy_side"),
                )
            elif self.trade_type == "Iron Condor":
                from .strategies.iron_condor import IronCondor

                strategy = IronCondor.create_iron_condor(
                    symbol=self.config.ticker,
                    put_long_strike=STRATEGY_PARAMS.get("put_long_strike"),
                    put_short_strike=STRATEGY_PARAMS.get("put_short_strike"),
                    call_short_strike=STRATEGY_PARAMS.get("call_short_strike"),
                    call_long_strike=STRATEGY_PARAMS.get("call_long_strike"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                )
            elif self.trade_type == "Iron Butterfly":
                from .strategies.iron_butterfly import IronButterfly

                strategy = IronButterfly.create_iron_butterfly(
                    symbol=self.config.ticker,
                    lower_strike=STRATEGY_PARAMS.get("lower_strike"),
                    middle_strike=STRATEGY_PARAMS.get("middle_strike"),
                    upper_strike=STRATEGY_PARAMS.get("upper_strike"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    strategy_side=STRATEGY_PARAMS.get("strategy_side"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                )
            elif self.trade_type == "Straddle":
                from .strategies.straddle import Straddle

                strategy = Straddle.create_straddle(
                    symbol=self.config.ticker,
                    strike=STRATEGY_PARAMS.get("strike"),
                    expiration=STRATEGY_PARAMS.get("dte"),
                    contracts=STRATEGY_PARAMS.get("contracts"),
                    commission=STRATEGY_PARAMS.get("commission", 0.5),
                    max_extra_days=STRATEGY_PARAMS.get("max_extra_days", None),
                    entry_time=bar,
                    option_chain_df=option_chain_df,
                    exit_scheme=STRATEGY_PARAMS.get("exit_scheme"),
                    strategy_side=STRATEGY_PARAMS.get("strategy_side"),
                )
            else:
                logger.warning(f"Unsupported trade type: {self.trade_type}")
                return

            order = self.option_broker.create_order(strategy)
            self.last_update_time = bar.round("15min")
            logger.info(f"{bar}: Created order: {order}")
            if self.add_order(order):
                logger.info(f"{bar}: Added order: {order}")
            else:
                logger.info(f"{bar}: Order not added {order}")
