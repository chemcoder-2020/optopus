from .base_trade_manager import BaseTradeManager
from ..brokers.broker import OptionBroker
from ..brokers.order import Order
from datetime import datetime
from typing import List, Optional, Any
import numpy as np
import pandas as pd
from loguru import logger


class LiveTradeManager(BaseTradeManager):
    """
    Trade manager for live trading, inheriting shared logic.
    """

    def __init__(self, config, name="LiveTradeManager"):
        super().__init__(config)
        self.option_broker = OptionBroker(config)
        self.name = name
        self.automation_on = True
        self.management_on = True
        self.performance_data = []
        self.trades_entered_today = 0
        self.trades_entered_this_week = 0
        self.active_orders: List[Order] = self.active_trades
        self.closed_orders: List[Order] = self.closed_trades

    def fetch_data(self):
        """
        Fetch live option chain data.
        """
        try:
            option_chain_df = self.option_broker.data.get_option_chain(
                self.config.ticker
            )
            return option_chain_df
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return None

    def create_trade(self, params, data):
        """
        Create a live order object.
        """
        # Placeholder: orchestrator or strategy should create Order
        return None

    def submit_order(self, trade):
        """
        Submit order to broker.
        """
        try:
            if trade.submit_entry():
                return True
            else:
                logger.warning(f"Order submission failed: {trade}")
                return False
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False

    def evaluate_new_entries(self, data, params):
        """
        Evaluate entry conditions and add new live orders.
        """
        # Placeholder: orchestrator or strategy should override or call this
        pass

    def _update_trade_counts(self):
        """
        Update trade counts for today and this week.
        """
        if self.last_update_time:
            self.trades_entered_today = sum(
                1
                for t in self.active_trades + self.closed_trades
                if getattr(t, "entry_time", None)
                and t.entry_time.date() == self.last_update_time.date()
            )
            self.trades_entered_this_week = sum(
                1
                for t in self.active_trades
                if getattr(t, "entry_time", None)
                and t.entry_time.isocalendar()[1]
                == self.last_update_time.isocalendar()[1]
            )

    def _record_performance_data(self, current_time, data):
        """
        Record performance data at the current time.
        """
        total_pl = self.get_total_pl()
        closed_pl = self.get_closed_pl()
        active_positions = len(self.active_trades)
        underlying_last = (
            data["UNDERLYING_LAST"].iloc[0]
            if isinstance(data, pd.DataFrame) and "UNDERLYING_LAST" in data.columns
            else 0
        )
        self.performance_data.append(
            {
                "time": current_time,
                "total_pl": total_pl,
                "closed_pl": closed_pl,
                "underlying_last": underlying_last,
                "active_positions": active_positions,
                "indicators": {**self.context.get("indicators", {})},
            }
        )

    def get_total_pl(self):
        """
        Calculate total P&L.
        """
        return np.nansum(
            [
                getattr(t, "filter_pl", 0)
                for t in self.active_trades + self.closed_trades
            ]
        )

    def get_closed_pl(self):
        """
        Calculate closed P&L.
        """
        return np.nansum([getattr(t, "filter_pl", 0) for t in self.closed_trades])

    def get_performance_metrics(self):
        """
        Placeholder for performance metrics calculation.
        """
        # Can be extended with Sharpe, drawdown, etc.
        return {}

    def auth_refresh(self):
        """
        Refresh authentication tokens.
        """
        try:
            self.option_broker.auth.refresh_access_token()
        except Exception:
            logger.exception("Attempting full authentication flow")
            self.option_broker.auth.authenticate()

        for order in self.active_orders + self.closed_orders:
            order.auth = self.option_broker.auth

        self.option_broker.data.auth = self.option_broker.auth
        self.option_broker.trading.auth = self.option_broker.auth

    def authenticate(self):
        """
        Perform full authentication.
        """
        self.option_broker.auth.authenticate()

        for order in self.active_orders + self.closed_orders:
            order.auth = self.option_broker.auth

        self.option_broker.data.auth = self.option_broker.auth
        self.option_broker.trading.auth = self.option_broker.auth

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.
        """
        for i, order in enumerate(self.active_orders):
            if order.order_id.split("/")[-1] == order_id:
                order.cancel()
                self.active_orders.pop(i)
                self._update_trade_counts()
                return True
        logger.warning(f"Order with ID {order_id} not found.")
        return False

    def close_order(self, order_id: str) -> bool:
        """
        Close an order by ID.
        """
        for order in self.active_orders:
            if order.order_id.split("/")[-1] == order_id:
                order.close_order()
                self.closed_orders.append(order)
                self.active_orders.remove(order)
                self.available_to_trade += order.get_required_capital()
                self._update_trade_counts()
                return True
        logger.warning(f"Order with ID {order_id} not found.")
        return False

    def liquidate_all(self):
        """
        Close all active orders.
        """
        for order in self.active_orders[:]:
            self.close_order(order.order_id.split("/")[-1])
        self._update_trade_counts()
