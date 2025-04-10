from .base_trade_manager import BaseTradeManager

# from datetime import datetime  # Removed unused import
import numpy as np
import pandas as pd

# from loguru import logger  # Removed unused import


class BacktestTradeManager(BaseTradeManager):
    """
    Trade manager for backtesting, inheriting shared logic.
    """

    def __init__(self, config):
        super().__init__(config)
        self.performance_data = []
        self.trades_entered_today = 0
        self.trades_entered_this_week = 0

    def fetch_data(self):
        """
        Fetch historical data (to be implemented by orchestrator).
        """
        # In backtesting, data is usually passed externally
        return None

    def create_trade(self, params, data):
        """
        Create a simulated option trade.
        """
        # Placeholder: orchestrator or strategy should create OptionStrategy
        return None

    def submit_order(self, trade):
        """
        No-op for backtesting (simulate order submission).
        """
        return True

    def evaluate_new_entries(self, data, params):
        """
        Evaluate entry conditions and add new simulated trades.
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
