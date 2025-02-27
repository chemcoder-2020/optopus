from .base import EntryConditionChecker
from loguru import logger
import pandas as pd


class TrailingStopEntry(EntryConditionChecker):
    def __init__(self, **kwargs):
        """
        Trailing stop entry condition checker.

        Args:
            trailing_entry_direction (str): 'bullish' or 'bearish' direction for trailing stop
            trailing_entry_threshold (float): Threshold value for entry trigger
            method (str): Calculation method ('percent', 'dollar', or 'atr')
            trailing_entry_reset_period (str): Reset period for trailing stop ('D', 'W', 'M', 'Y')
        """
        self.trailing_entry_direction = kwargs.get(
            "trailing_entry_direction", "bullish"
        ).lower()
        if self.trailing_entry_direction not in ("bullish", "bearish"):
            raise ValueError("trailing_entry_direction must be 'bullish' or 'bearish'")

        self.trailing_entry_threshold = abs(kwargs.get("trailing_entry_threshold", 1.0))
        self.method = kwargs.get("method", "percent").lower()
        self.trailing_entry_reset_period = kwargs.get(
            "trailing_entry_reset_period", None
        )

        # State tracking variables
        self.current_date = None
        self.current_week = None
        self.current_month = None
        self.current_year = None
        self.cum_min = None  # Track cumulative min price
        self.cum_max = None  # Track cumulative max price

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        if self.trailing_entry_reset_period is not None:
            if self.trailing_entry_reset_period == "D":
                if self.current_date != time.date():
                    self.current_date = time.date()
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "W":
                current_week = time.isocalendar()[1]  # Get ISO week number
                if self.current_week != current_week:
                    self.current_week = current_week
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "M":
                if self.current_month != time.month:
                    self.current_month = time.month
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "Y":
                if self.current_year != time.year:
                    self.current_year = time.year
                    self.cum_min = None
                    self.cum_max = None

        current_price = strategy.underlying_last

        if self.cum_max is None:
            self.cum_max = current_price

        if self.cum_min is None:
            self.cum_min = current_price

        # Update cumulative min/max
        self.cum_min = min(self.cum_min, current_price)
        self.cum_max = max(self.cum_max, current_price)

        # Calculate price movement from extreme
        if self.trailing_entry_direction == "bullish":
            if self.method == "dollar":
                price_change = current_price - self.cum_min
            elif self.method == "percent":
                price_change = ((current_price - self.cum_min) / self.cum_min) * 100
            elif self.method == "atr":
                if hasattr(manager, "atr") and manager.atr > 0:
                    price_change = (current_price - self.cum_min) / manager.atr
                else:
                    logger.error(
                        "ATR not available or invalid for trailing stop calculation"
                    )
                    return False
        else:
            if self.method == "dollar":
                price_change = current_price - self.cum_max
            elif self.method == "percent":
                price_change = ((current_price - self.cum_max) / self.cum_max) * 100
            elif self.method == "atr":
                if hasattr(manager, "atr") and manager.atr > 0:
                    price_change = (current_price - self.cum_max) / manager.atr
                else:
                    logger.error(
                        "ATR not available or invalid for trailing stop calculation"
                    )
                    return False

        if self.trailing_entry_direction == "bullish":
            return price_change >= self.trailing_entry_threshold
        else:
            return price_change <= -self.trailing_entry_threshold
