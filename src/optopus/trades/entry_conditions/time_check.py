from .base import EntryConditionChecker
from datetime import datetime
from typing import TYPE_CHECKING, Union, List, Tuple
from loguru import logger
import pandas as pd

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


class TimeBasedEntryCondition(EntryConditionChecker):
    """
    Checks if current time is within allowed entry times.

    Args:
        allowed_times (List[Union[Tuple[str, str], str]]): List of time ranges in "HH:MM-HH:MM" format (default: 9:45AM-3:45PM ET)
        allowed_days (List[Union[int, str]]): List of allowed days (0=Mon-6=Sun or 'Mon'-'Sun', default: Mon-Fri)
        timezone (str): Timezone string (default: "America/New_York")
    """

    def __init__(
        self,
        allowed_times: List[Union[Tuple[str, str], str]] = ["09:45-15:45"],
        allowed_days: List[Union[int, str]] = ["Mon", "Tue", "Wed", "Thu", "Fri"],
        timezone: str = "America/New_York",
    ):
        self.allowed_times = allowed_times
        self.allowed_days = allowed_days
        self.timezone = timezone

        # Convert string day names to numbers if needed
        self.allowed_day_numbers = []
        day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        for day in self.allowed_days:
            if isinstance(day, str):
                self.allowed_day_numbers.append(day_map[day.lower()[:3]])
            else:
                self.allowed_day_numbers.append(int(day))

        # Parse time ranges
        self.time_ranges = []
        for time_spec in self.allowed_times:
            if isinstance(time_spec, str):
                start_str, end_str = time_spec.split("-")
            else:
                start_str, end_str = time_spec

            start = pd.Timestamp(start_str)  # .tz_localize(self.timezone)
            end = pd.Timestamp(end_str)  # .tz_localize(self.timezone)
            self.time_ranges.append((start.time(), end.time()))

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        current_time = pd.Timestamp(time)  # .tz_convert(self.timezone)

        # Check allowed days
        if self.allowed_day_numbers:
            day_of_week = current_time.dayofweek
            if day_of_week not in self.allowed_day_numbers:
                logger.info(
                    f"Entry not allowed on {current_time.day_name()}. Rejecting entry on TimeBasedEntryCondition"
                )
                return False

        # Check allowed time ranges
        if self.time_ranges:
            in_window = False
            curr_time = current_time.time()
            for start, end in self.time_ranges:
                if start <= curr_time <= end:
                    in_window = True
                    break

            if not in_window:
                logger.info(
                    f"Current time {curr_time.strftime('%H:%M:%S')} not in allowed ranges. Rejecting entry on TimeBasedEntryCondition"
                )
                return False

        logger.info("Time-based entry conditions met")
        return True
