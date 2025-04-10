import abc
from datetime import datetime
from typing import List, Optional, Any

import numpy as np
# import pandas as pd  # Removed unused import
from loguru import logger



class BaseTradeManager(abc.ABC):
    """
    Abstract base class for unified trade management.
    """

    def __init__(self, config):
        self.config = config
        self.capital = config.initial_capital
        self.allocation = config.initial_capital
        self.available_to_trade = config.initial_capital
        self.active_trades: List[Any] = []
        self.closed_trades: List[Any] = []
        self.last_update_time: Optional[datetime] = None
        self.context = {}

    def next(self, data=None, params=None):
        """
        Template method: update existing trades/orders,
        then evaluate new entries.
        """
        if data is None:
            data = self.fetch_data()
        self.update_trades(data)
        self.evaluate_new_entries(data, params)

    @abc.abstractmethod
    def fetch_data(self):
        """
        Fetch market data (historical or live).
        """
        pass

    @abc.abstractmethod
    def create_trade(self, params, data):
        """
        Create a trade or order object.
        """
        pass

    @abc.abstractmethod
    def submit_order(self, trade):
        """
        Submit order to broker or simulate entry.
        """
        pass

    def update_trades(self, data):
        """
        Update all active trades/orders with new data.
        """
        to_close = []
        for trade in list(self.active_trades):
            try:
                updated = trade.update(datetime.now(), data)
                if not updated:
                    logger.warning(f"Trade {trade} update failed.")
                elif getattr(trade, "status", "") == "CLOSED":
                    to_close.append(trade)
            except Exception as e:
                logger.error(f"Error updating trade {trade}: {e}")
        for trade in to_close:
            self.close_trade(trade)
        self._update_trade_counts()
        self._record_performance_data(datetime.now(), data)

    def evaluate_new_entries(self, data, params):
        """
        Evaluate entry conditions and add new trades/orders if conditions met.
        """
        # Placeholder: subclasses or strategies override this
        pass

    def add_trade(self, trade):
        """
        Add a new trade/order to active list and update capital.
        """
        try:
            if not self._can_add_trade(trade):
                logger.info(
                    f"Rejecting trade {trade} - entry conditions not met"
                )
                return False
            required_capital = trade.get_required_capital()
            trade.entry_delta = getattr(trade, "current_delta", lambda: None)()
            self.active_trades.append(trade)
            self.available_to_trade -= required_capital
            self._update_trade_counts()
            logger.success(
                f"Added trade {trade} (Required: ${required_capital:.2f}, "
                f"Available: ${self.available_to_trade:.2f})"
            )
            return True
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False

    def close_trade(self, trade):
        """
        Close a trade/order, update capital and allocation.
        """
        try:
            self.active_trades.remove(trade)
            self.closed_trades.append(trade)
            pl_change = getattr(trade, "filter_pl", 0)
            if np.isnan(pl_change):
                pl_change = 0
            recovered_capital = trade.get_required_capital()
            self.capital += pl_change
            self.available_to_trade += recovered_capital
            if self.config.gain_reinvesting:
                new_allocation = max(self.capital, self.allocation)
                added_allocation = new_allocation - self.allocation
                self.allocation = new_allocation
                self.available_to_trade += added_allocation
            self._update_trade_counts()
        except Exception as e:
            logger.error(f"Error closing trade: {e}")

    def _can_add_trade(self, trade):
        """
        Check if a new trade/order can be added based on capital
        and entry conditions.
        """
        if self.capital <= 0:
            logger.warning("No capital left to add trade.")
            return False
        # Additional entry conditions can be added here or overridden
        return True

    def _update_trade_counts(self):
        """
        Update trade counts (today, week, etc.).
        """
        # Placeholder: subclasses can override
        pass

    def _record_performance_data(self, current_time, data):
        """
        Record performance data at the current time.
        """
        # Placeholder: subclasses can override
        pass

    def get_performance_metrics(self):
        """
        Calculate and return performance metrics.
        """
        # Placeholder: subclasses can override
        return {}
