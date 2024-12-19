import pandas as pd
from pandas import Timestamp, Timedelta
from .option_leg import OptionLeg
import datetime
import numpy as np
import time  # Added for performance profiling
import logging  # Added for logging
from loguru import logger
import cProfile
import pstats
from pstats import SortKey
from .exit_conditions import DefaultExitCondition, ExitConditionChecker
from .option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional


class OptionStrategy:
    """
    Represents an options trading strategy composed of multiple option legs.

    Attributes:
        symbol (str): The underlying asset symbol.
        strategy_type (str): The type of option strategy.
        legs (List[OptionLeg]): List of OptionLeg objects in the strategy.
        entry_time (Timestamp): The entry time for the strategy.
        current_time (Timestamp): The current time.
        status (str): The status of the strategy ("OPEN" or "CLOSED").
        profit_target (float): The profit target percentage.
        stop_loss (float): The stop loss percentage.
        trailing_stop (float): The trailing stop percentage.
        highest_return (float): The highest return percentage achieved.
        entry_net_premium (float): The net premium at entry.
        net_premium (float): The current net premium.
        current_bid (float): The current bid price of the strategy.
        current_ask (float): The current ask price of the strategy.
        won (bool or None): Whether the trade was won (True), lost (False), or is still open (None).
        DIT (int): Days in Trade, representing the number of calendar days since the trade was opened.
        contracts (int): The number of contracts for the strategy.
        commission (float): The commission per contract.
        exit_scheme (ExitConditionChecker): The exit condition scheme to use.

    Methods:
        add_leg(leg, ratio): Add an option leg to the strategy with a specified ratio.
        update(current_time, option_chain_df): Update the strategy with new market data and check exit conditions.
        close_strategy(close_time, option_chain_df): Close the option strategy.
        total_pl(): Calculate the total profit/loss of the strategy.
        return_percentage(): Calculate the return percentage of the strategy.
        current_delta(): Calculate the current delta of the strategy.
        conflicts_with(other_spread): Check if this option spread conflicts with another option spread.
        get_required_capital(): Calculate the required capital for the option strategy.
        get_required_capital_per_contract(): Calculate the required capital per contract for the option strategy.
        calculate_net_premium(): Calculate the net premium based on current prices and position sides of the legs.
        calculate_bid_ask(): Calculate the bid-ask spread for the entire option strategy.
        return_over_risk(): Calculate the current return over risk value for the spread.
    """

    def __init__(
        self,
        symbol: str,
        strategy_type: str,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        contracts: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Initialize an OptionStrategy object.

        Args:
            symbol (str): The underlying asset symbol.
            strategy_type (str): The type of option strategy.
            profit_target (float, optional): The profit target percentage. Defaults to None.
            stop_loss (float, optional): The stop loss percentage. Defaults to None.
            trailing_stop (float, optional): The trailing stop percentage. Defaults to None.
            contracts (int, optional): The number of contracts for the strategy. Defaults to 1.
            commission (float, optional): Commission per contract. Defaults to 0.5.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

        Returns:
            None

        Raises:
            ValueError: If the exit_scheme is not an instance of ExitConditionChecker.
        """
        if exit_scheme is not None and not isinstance(
            exit_scheme, ExitConditionChecker
        ):
            raise ValueError("exit_scheme must be an instance of ExitConditionChecker")

        self.symbol = symbol
        self.strategy_type = strategy_type
        self.strategy_side = None
        self.legs = []
        self.entry_time = None
        self.exit_time = None
        self.current_time = None
        self.status = "OPEN"
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.highest_return = 0
        self.entry_net_premium = None
        self.net_premium = None
        self.current_bid = None
        self.current_ask = None
        self.won = None  # Initialize won to None for open trades
        self.DIT = 0  # Initialize Days in Trade to 0
        self._contracts = contracts
        self.commission = commission
        self.leg_ratios = []  # Store the ratio for each leg
        self.entry_ror = None
        self.exit_net_premium = None
        self.exit_ror = None
        self.entry_underlying_last = None
        self.exit_underlying_last = None
        self.exit_dit = None
        self.exit_dte = None
        self.exit_scheme = exit_scheme

    @staticmethod
    def _standardize_time(time_value):
        """Convert time to a pandas Timestamp object.

        Args:
            time_value (str, pd.Timestamp, pd.DatetimeIndex, datetime.datetime): The time value to standardize.

        Returns:
            pd.Timestamp: The standardized Timestamp object.

        Raises:
            ValueError: If the time format is unsupported.
        """
        if isinstance(time_value, str):
            return pd.to_datetime(time_value)
        elif isinstance(
            time_value, (pd.Timestamp, pd.DatetimeIndex, datetime.datetime)
        ):
            return pd.Timestamp(time_value)
        else:
            raise ValueError(f"Unsupported time format: {type(time_value)}")

    @property
    def contracts(self):
        """
        Get the number of contracts in the strategy.

        Returns:
            int: The number of contracts.
        """
        return self._contracts

    @contracts.setter
    def contracts(self, value):
        """
        Set the number of contracts for the strategy and update all legs accordingly.

        Args:
            value (int): The new number of contracts.
        """
        self._contracts = value
        self._update_leg_contracts()

    def _update_leg_contracts(self):
        """
        Update the contracts for all legs based on their ratios and the strategy's contract count.
        This is an internal method called when the strategy's contract count is modified.
        """
        for leg, ratio in zip(self.legs, self.leg_ratios):
            leg.contracts = self._contracts * ratio

    def add_leg(self, leg: OptionLeg, ratio: int = 1):
        """
        Add an option leg to the strategy with a specified ratio.

        Args:
            leg (OptionLeg): The option leg to add.
            ratio (int): The ratio of this leg relative to the strategy's contract count.

        Raises:
            ValueError: If the leg's entry time doesn't match the strategy's entry time.
        """
        if not self.legs:
            self.entry_time = self._standardize_time(leg.entry_time)
            self.current_time = (
                self.entry_time
            )  # Set current_time to entry_time initially
            self.DIT = 0  # Reset DIT when adding the first leg
            self.entry_underlying_last = leg.entry_underlying_last
        elif self._standardize_time(leg.entry_time) != self.entry_time:
            raise ValueError("All legs must have the same entry time")

        self.legs.append(leg)
        self.leg_ratios.append(ratio)
        leg.contracts = self._contracts * ratio

    def update(
        self,
        current_time: Union[str, Timestamp, datetime.datetime],
        option_chain_df: pd.DataFrame,
    ):
        """
        Update the strategy with new market data and check exit conditions. This method will close the strategy if the exit conditions are met.

        Args:
            current_time (Union[str, Timestamp, datetime.datetime]): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
        """
        if self.status == "CLOSED":
            return  # Do nothing if the strategy is already closed

        self.current_time = self._standardize_time(current_time)

        # Update DIT
        if self.entry_time:
            self.DIT = (self.current_time.date() - self.entry_time.date()).days

        # Update the current attributes of each leg
        for leg in self.legs:
            leg.update(current_time, option_chain_df)

        new_net_premium = self.calculate_net_premium()

        self.net_premium = new_net_premium

        if self.status == "OPEN":
            self._check_exit_conditions(option_chain_df)

        # Update exit_dte
        if self.status == "OPEN":
            self.exit_dte = (
                pd.to_datetime(self.legs[0].expiration) - self.current_time
            ).days

        # Calculate and store the strategy's bid-ask spread
        self.current_bid, self.current_ask = self.calculate_bid_ask()
        return True

    def update_entry_net_premium(self):
        """Update the entry net premium. Helpful to update entry net premium after actual trading order is filled."""
        _replace_premium = 0
        for leg, ratio in zip(self.legs, self.leg_ratios):
            premium_adjustment = leg.entry_price * ratio
            if leg.position_side == "SELL":
                _replace_premium += premium_adjustment
            else:  # BUY
                _replace_premium -= premium_adjustment

        self.entry_net_premium = _replace_premium

    def update_exit_net_premium(self):
        """Update the exit net premium. Helpful to update exit net premium after actual trading order is filled."""
        _replace_premium = 0
        for leg, ratio in zip(self.legs, self.leg_ratios):
            premium_adjustment = leg.exit_price * ratio
            if leg.position_side == "SELL":
                _replace_premium += premium_adjustment
            else:  # BUY
                _replace_premium -= premium_adjustment

        self.exit_net_premium = _replace_premium

    def _check_exit_conditions(self, option_chain_df):
        """Check and apply exit conditions."""
        current_return = self.return_percentage()

        # Update highest return for trailing stop
        if hasattr(self, "median_return_percentage"):
            self.highest_return = max(
                self.highest_return, self.median_return_percentage
            )
        else:
            self.highest_return = max(self.highest_return, current_return)

        if hasattr(self, "exit_scheme") and self.exit_scheme:
            if self.exit_scheme.should_exit(self, self.current_time, option_chain_df):
                self._close_strategy(option_chain_df)
                return
        else:
            # Check profit target
            if self.profit_target and current_return >= self.profit_target:
                self._close_strategy(option_chain_df)
                return

            # Check stop loss
            if self.stop_loss and current_return <= -self.stop_loss:
                self._close_strategy(option_chain_df)
                return

            # Check trailing stop
            if (
                self.trailing_stop
                and (self.highest_return - current_return) >= self.trailing_stop
            ):
                self._close_strategy(option_chain_df)
                return

            # Check time-based conditions
            try:
                expiration_date = pd.Timestamp(self.legs[0].expiration).date()
                expiration_datetime = pd.Timestamp.combine(
                    expiration_date, pd.Timestamp("16:00:00").time()
                )

                # Close if within 15 minutes of expiration
                if self.current_time >= expiration_datetime - Timedelta(minutes=15):
                    self._close_strategy(option_chain_df)
                    return

                # Close if past expiration
                if self.current_time > expiration_datetime:
                    self._close_strategy(option_chain_df)
                    logger.warning(
                        f"Option strategy has expired before: {self.legs[0].expiration}"
                    )
                    return
            except (AttributeError, ValueError, TypeError) as e:
                print(f"Error in time-based exit check: {e}")
                print(f"Current time: {self.current_time}")
                print(f"Leg expiration: {self.legs[0].expiration}")

    def _close_strategy(self, option_chain_df):
        if self.status == "CLOSED":
            return  # Already closed, do nothing

        self.status = "CLOSED"
        self.won = self.total_pl() > 0
        self.exit_time = self.current_time
        self.exit_net_premium = (
            min(self.net_premium, self.max_exit_net_premium)
            if hasattr(self, "max_exit_net_premium")
            else self.net_premium
        )
        self.exit_ror = self.return_over_risk()
        self.exit_underlying_last = self.legs[0].underlying_last
        self.exit_dit = self.DIT
        self.exit_dte = (
            pd.to_datetime(self.legs[0].expiration).date() - self.exit_time.date()
        ).days

    def _reopen_strategy(self):
        """
        Reopen a closed strategy by resetting all exit-related attributes.
        This is an internal method used when a strategy needs to be reopened.
        """
        self.status = "OPEN"
        self.won = None
        self.exit_time = None
        self.exit_net_premium = None
        self.exit_ror = None
        self.exit_underlying_last = None
        self.exit_dit = None
        self.exit_dte = None
        self.highest_return = max(0, self.return_percentage())

    def close_strategy(
        self,
        close_time: Union[str, Timestamp, datetime.datetime],
        option_chain_df: pd.DataFrame,
    ):
        """
        Close the option strategy.

        Args:
            close_time (Union[str, Timestamp, datetime.datetime]): The time at which the strategy is being closed.
            option_chain_df (pd.DataFrame): The option chain data at close time.

        Raises:
            ValueError: If the strategy is already closed.
        """
        if self.status == "CLOSED":
            raise ValueError("Strategy is already closed")

        self.update(close_time, option_chain_df)
        self._close_strategy(option_chain_df)

    def calculate_total_commission(self):
        return sum(leg.calculate_total_commission() for leg in self.legs)

    def total_pl(self) -> float:
        """Calculate the total profit/loss of the strategy.

        Returns:
            float: The total profit/loss of the strategy.
        """

        if hasattr(self, "strategy_side") and self.strategy_side == "CREDIT":
            return (
                (self.entry_net_premium - self.calculate_net_premium())
                * 100
                * self.contracts
            ) - self.calculate_total_commission()
        elif hasattr(self, "strategy_side") and self.strategy_side == "DEBIT":
            return (
                (self.calculate_net_premium() - self.entry_net_premium)
                * 100
                * self.contracts
            ) - self.calculate_total_commission()
        else:
            raise ValueError(f"Unsupported strategy side: {self.strategy_side}")

    def return_percentage(self) -> float:
        """Calculate the return percentage of the strategy.

        Returns:
            float: The return percentage of the strategy.
        """
        premium = abs(self.entry_net_premium)
        if premium == 0:
            return 0
        return (self.total_pl() / (premium * 100 * self.contracts)) * 100

    def current_delta(self) -> float:
        """
        Calculate the current delta of the strategy.
        Takes into account position side (BUY/SELL) and contract multiplier.

        Returns:
            float: The current delta of the strategy.
        """
        return sum(
            leg.current_delta
            * leg.contracts
            * 100
            * (1 if leg.position_side == "BUY" else -1)
            for leg in self.legs
        )

    def conflicts_with(self, other_spread) -> bool:
        """
        Check if this option spread conflicts with another option spread.

        Args:
            other_spread (OptionStrategy): The other option spread to compare with.

        Returns:
            bool: True if there's a conflict, False otherwise.
        """
        for leg in self.legs:
            for other_leg in other_spread.legs:
                if leg.conflicts_with(other_leg):
                    return True
        return False

    @staticmethod
    def _get_strike(
        symbol: str,
        option_chain_df: pd.DataFrame,
        strike_selector,
        option_type: str,
        reference_strike=None,
        expiration=None,  # Add expiration as a parameter
    ):
        return OptionChainConverter.get_strike(
            symbol,
            option_chain_df,
            strike_selector,
            option_type,
            reference_strike,
            expiration,
        )

    @staticmethod
    def _get_expiration(
        option_chain_df: pd.DataFrame, expiration_input, entry_time: str
    ):
        return OptionChainConverter.get_expiration(
            option_chain_df, expiration_input, entry_time
        )

    @classmethod
    def get_strike_value(
        cls,
        converter,
        strike_input,
        expiration_date,
        option_type,
        reference_strike=None,
    ):
        if isinstance(strike_input, (int, float)):
            if abs(strike_input) < 1:
                # Numeric input treated as delta if float < 1
                return converter.get_desired_strike(
                    expiration_date, option_type, strike_input, by="delta"
                )
            else:
                # Numeric input treated as a specific strike price
                return converter.get_desired_strike(
                    expiration_date, option_type, strike_input, by="strike"
                )
        elif isinstance(strike_input, str):
            if strike_input.startswith(("+", "-")):
                try:
                    offset = float(strike_input)
                    if abs(offset) < 1:
                        # ATM relative strike with delta
                        return converter.get_desired_strike(
                            expiration_date, option_type, offset, by="delta"
                        )
                    else:
                        return converter.get_desired_strike(
                            expiration_date,
                            option_type,
                            reference_strike + offset,
                            by="strike",
                        )
                except ValueError:
                    raise ValueError(f"Invalid strike input: {strike_input}")
            else:
                try:
                    strike_price = float(strike_input)
                    return converter.get_desired_strike(
                        expiration_date, option_type, strike_price, by="strike"
                    )
                except ValueError:
                    if strike_input.upper() == "ATM":
                        return converter.get_atm_strike(expiration_date)
                    elif "ATM" in strike_input:
                        try:
                            offset = float(strike_input[3:].replace("%", ""))
                            if "%" in strike_input:
                                return converter.get_desired_strike(
                                    expiration_date,
                                    option_type,
                                    offset,
                                    by="atm_percent",
                                )
                            else:
                                return converter.get_desired_strike(
                                    expiration_date,
                                    option_type,
                                    offset,
                                    by="atm",
                                )
                        except ValueError:
                            raise ValueError(f"Invalid strike input: {strike_input}")
                    else:
                        raise ValueError(f"Invalid strike input: {strike_input}")
        else:
            raise ValueError(f"Unsupported strike input type: {type(strike_input)}")


    @classmethod
    def create_iron_condor(
        cls,
        symbol: str,
        put_long_strike,
        put_short_strike,
        call_short_strike,
        call_long_strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Create an iron condor option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            put_long_strike: The long put strike price or selector.
            put_short_strike: The short put strike price or selector.
            call_short_strike: The short call strike price or selector.
            call_long_strike: The long call strike price or selector.
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            leg_ratio (int, optional): The ratio of leg contracts to the strategy's contract count.
            commission (float, optional): Commission per contract per leg.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
            OptionStrategy: A vertical spread strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Vertical Spread",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        # Get strike prices using the converter
        short_strike_value = strategy.get_strike_value(
            converter, short_strike, expiration_date, option_type
        )

        long_strike_value = strategy.get_strike_value(
            converter,
            long_strike,
            expiration_date,
            option_type,
            reference_strike=(
                short_strike_value
                if isinstance(long_strike, str)
                and (long_strike[0] == "+" or long_strike[0] == "-")
                else None
            ),
        )

        if (long_strike_value > short_strike_value and option_type == "PUT") or (
            long_strike_value < short_strike_value and option_type == "CALL"
        ):
            strategy.strategy_side = "DEBIT"
        elif (long_strike_value < short_strike_value and option_type == "PUT") or (
            long_strike_value > short_strike_value and option_type == "CALL"
        ):
            strategy.strategy_side = "CREDIT"
        else:
            raise ValueError("Long and short strike values cannot be equal.")

        long_leg = OptionLeg(
            symbol,
            option_type,
            long_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )
        short_leg = OptionLeg(
            symbol,
            option_type,
            short_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "SELL",
            commission=commission,
        )

        strategy.add_leg(long_leg, leg_ratio)
        strategy.add_leg(short_leg, leg_ratio)

        # Calculate the width of the spread
        spread_width = abs(long_strike_value - short_strike_value)
        strategy.max_exit_net_premium = spread_width

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_dte = (
            pd.to_datetime(strategy.legs[0].expiration).date()
            - strategy.entry_time.date()
        ).days
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()
        strategy.entry_bid, strategy.entry_ask = (
            strategy.current_bid,
            strategy.current_ask,
        )

        if strategy.entry_net_premium > spread_width:
            raise ValueError(
                "Entry net premium cannot be greater than the spread width."
            )

        return strategy

    @classmethod
    def create_iron_condor(
        cls,
        symbol: str,
        put_long_strike,
        put_short_strike,
        call_short_strike,
        call_long_strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Create an iron condor option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            put_long_strike: The long put strike price or selector.
            put_short_strike: The short put strike price or selector.
            call_short_strike: The short call strike price or selector.
            call_long_strike: The long call strike price or selector.
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            leg_ratio (int, optional): The ratio of leg contracts to the strategy's contract count.
            commission (float, optional): Commission per contract per leg.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

        Returns:
            OptionStrategy: An iron condor strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Iron Condor",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        put_short_strike_value = strategy.get_strike_value(
            converter, put_short_strike, expiration_date, "PUT"
        )
        put_long_strike_value = strategy.get_strike_value(
            converter,
            put_long_strike,
            expiration_date,
            "PUT",
            reference_strike=(
                put_short_strike_value
                if isinstance(put_long_strike, str)
                and (put_long_strike[0] == "+" or put_long_strike[0] == "-")
                else None
            ),
        )

        # Get call strikes
        call_short_strike_value = strategy.get_strike_value(
            converter, call_short_strike, expiration_date, "CALL"
        )
        call_long_strike_value = strategy.get_strike_value(
            converter,
            call_long_strike,
            expiration_date,
            "CALL",
            reference_strike=(
                call_short_strike_value
                if isinstance(call_long_strike, str)
                and (call_long_strike[0] == "+" or call_long_strike[0] == "-")
                else None
            ),
        )

        if (
            put_long_strike_value > put_short_strike_value
            and call_short_strike_value > call_long_strike_value
        ):
            strategy.strategy_side = "DEBIT"
        elif (
            put_long_strike_value < put_short_strike_value
            and call_short_strike_value < call_long_strike_value
        ):
            strategy.strategy_side = "CREDIT"
        else:
            raise ValueError("Invalid Iron Condor strike values.")

        put_long_leg = OptionLeg(
            symbol,
            "PUT",
            put_long_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )
        put_short_leg = OptionLeg(
            symbol,
            "PUT",
            put_short_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "SELL",
            commission=commission,
        )
        call_short_leg = OptionLeg(
            symbol,
            "CALL",
            call_short_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "SELL",
            commission=commission,
        )
        call_long_leg = OptionLeg(
            symbol,
            "CALL",
            call_long_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )

        strategy.add_leg(put_long_leg, leg_ratio)
        strategy.add_leg(put_short_leg, leg_ratio)
        strategy.add_leg(call_short_leg, leg_ratio)
        strategy.add_leg(call_long_leg, leg_ratio)

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        if strategy.entry_net_premium > (
            abs(call_short_strike_value - call_long_strike_value)
            + abs(put_short_strike_value - put_long_strike_value)
        ):
            raise ValueError(
                "Entry net premium cannot be greater than the spread width."
            )

        strategy.max_exit_net_premium = max(
            abs(call_short_strike_value - call_long_strike_value),
            abs(put_short_strike_value - put_long_strike_value),
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_dte = (
            pd.to_datetime(strategy.legs[0].expiration).date()
            - strategy.entry_time.date()
        ).days
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    @classmethod
    def create_straddle(
        cls,
        symbol: str,
        strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        position_side="BUY",
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Create a straddle option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            strike: The strike price or selector for both legs.
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            leg_ratio (int, optional): The ratio of leg contracts to the strategy's contract count.
            commission (float, optional): Commission per contract per leg.
            position_side (str, optional): Position side ('BUY' or 'SELL'). Defaults to 'BUY'.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

        Returns:
            OptionStrategy: A straddle strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Straddle",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        # Get strike prices using the converter
        call_strike_value = cls.get_strike_value(
            converter, strike, expiration_date, "CALL"
        )
        put_strike_value = cls.get_strike_value(
            converter, call_strike_value, expiration_date, "PUT"
        )

        call_leg = OptionLeg(
            symbol,
            "CALL",
            call_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            position_side,
            commission=commission,
        )
        put_leg = OptionLeg(
            symbol,
            "PUT",
            put_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            position_side,
            commission=commission,
        )

        strategy.strategy_side = "DEBIT" if position_side == "BUY" else "CREDIT"

        strategy.add_leg(call_leg, leg_ratio)
        strategy.add_leg(put_leg, leg_ratio)

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_dte = (
            pd.to_datetime(strategy.legs[0].expiration).date()
            - strategy.entry_time.date()
        ).days
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    @classmethod
    def create_butterfly(
        cls,
        symbol: str,
        lower_strike,
        middle_strike,
        upper_strike,
        expiration,
        strategy_side: str,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Create a butterfly option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            lower_strike: The lower strike price, delta, or ATM offset (e.g., "-2", -0.3, or "ATM").
            middle_strike: The middle strike price, delta, or ATM offset.
            upper_strike: The upper strike price, delta, or ATM offset (e.g., "+2", 0.3, or "ATM").
            expiration (str or int): The option expiration date or target DTE.
            strategy_side (str): The strategy side, either "DEBIT" or "CREDIT".
            contracts (int): The number of contracts for the strategy (will be doubled for the middle leg).
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            commission (float, optional): Commission percentage.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

        Returns:
            OptionStrategy: A butterfly strategy object.
        """
        if strategy_side not in ["DEBIT", "CREDIT"]:
            raise ValueError("Invalid strategy side. Must be 'DEBIT' or 'CREDIT'.")

        if strategy_side == "DEBIT":
            strategy = cls.create_iron_condor(
                symbol=symbol,
                put_long_strike=middle_strike,
                put_short_strike=lower_strike,
                call_short_strike=upper_strike,
                call_long_strike=middle_strike,
                expiration=expiration,
                contracts=contracts,
                entry_time=entry_time,
                option_chain_df=option_chain_df,
                profit_target=profit_target,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                commission=commission,
                exit_scheme=exit_scheme,
            )
        else:
            strategy = cls.create_iron_condor(
                symbol=symbol,
                put_long_strike=lower_strike,
                put_short_strike=middle_strike,
                call_short_strike=middle_strike,
                call_long_strike=upper_strike,
                expiration=expiration,
                contracts=contracts,
                entry_time=entry_time,
                option_chain_df=option_chain_df,
                profit_target=profit_target,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                commission=commission,
                exit_scheme=exit_scheme,
            )
        strategy.strategy_type = "Butterfly"
        return strategy

    @classmethod
    def create_naked_call(
        cls,
        symbol: str,
        strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        commission: float = 0.5,
        strategy_side: str = "DEBIT",
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
    ):
        """
        Create a naked call option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            strike: The strike price, delta, or ATM offset (e.g., "+2", 0.3, or "ATM").
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            commission (float, optional): Commission percentage.
            strategy_side (str, optional): The strategy side ('DEBIT' or 'CREDIT'). Defaults to 'DEBIT'.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

        Returns:
            OptionStrategy: A naked call strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Naked Call",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        # Get strike price using the converter
        strike_value = cls.get_strike_value(converter, strike, expiration_date, "CALL")

        call_leg = OptionLeg(
            symbol,
            "CALL",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY" if strategy_side == "DEBIT" else "SELL",
            commission=commission,
        )

        strategy.strategy_side = strategy_side

        strategy.add_leg(call_leg)
        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_dte = (
            pd.to_datetime(strategy.legs[0].expiration).date()
            - strategy.entry_time.date()
        ).days
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    @classmethod
    def create_naked_put(
        cls,
        symbol: str,
        strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = DefaultExitCondition(
            profit_target=40,
            exit_time_before_expiration=Timedelta(minutes=15),
            window_size=5,
        ),
        strategy_side: str = "DEBIT",
    ):
        """
        Create a naked put option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            strike: The strike price, delta, or ATM offset (e.g., "-2", -0.3, or "ATM").
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            commission (float, optional): Commission percentage.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.
            strategy_side (str, optional): The strategy side ('DEBIT' or 'CREDIT'). Defaults to 'DEBIT'.

        Returns:
            OptionStrategy: A naked put strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Naked Put",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        # Get strike price using the converter
        strike_value = cls.get_strike_value(converter, strike, expiration_date, "PUT")

        put_leg = OptionLeg(
            symbol,
            "PUT",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY" if strategy_side == "DEBIT" else "SELL",
            commission=commission,
        )

        strategy.strategy_side = strategy_side

        strategy.add_leg(put_leg)
        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_dte = (
            pd.to_datetime(strategy.legs[0].expiration).date()
            - strategy.entry_time.date()
        ).days
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    def get_required_capital(self) -> float:
        """
        Calculate the required capital for the option strategy.
        This is typically the maximum potential loss of the strategy.

        Returns:
            float: The required capital for the strategy, including commission.
        """
        if self.strategy_side == "CREDIT":
            if self.strategy_type in ["Vertical Spread", "Iron Condor", "Butterfly"]:
                max_width = max(
                    abs(leg1.strike - leg2.strike)
                    for leg1, leg2 in zip(self.legs[::2], self.legs[1::2])
                )
                required_capital = (
                    (max_width - self.entry_net_premium) * 100 * self.contracts
                )
            elif self.strategy_type == "Straddle":
                collateral = sum(leg.strike for leg in self.legs)
                required_capital = (
                    (collateral - self.entry_net_premium) * 100 * self.contracts
                )
            elif self.strategy_type in ["Naked Call", "Naked Put"]:
                required_capital = (
                    (self.legs[0].strike - self.entry_net_premium)
                    * 100
                    * self.contracts
                )
            else:
                raise ValueError(f"Unsupported strategy type: {self.strategy_type}")
        elif self.strategy_side == "DEBIT":
            required_capital = self.entry_net_premium * 100 * self.contracts
        return required_capital + self.calculate_total_commission()

    def get_required_capital_per_contract(self) -> float:
        """
        Calculate the required capital per contract for the option strategy.
        This is typically the maximum potential loss of the strategy per contract.

        Returns:
            float: The required capital per contract for the strategy, including commission.
        """

        required_capital_per_contract = self.get_required_capital() / self.contracts

        return required_capital_per_contract

    def calculate_net_premium(self) -> float:
        """
        Calculate the net premium based on the current prices and position sides of the legs.

        Returns:
            float: The calculated net premium, representing the average of the bid and ask prices.
        """
        bid, ask = self.calculate_bid_ask()
        logger.debug(f"bid: {bid}, ask: {ask}")
        net_premium = (bid + ask) / 2
        if net_premium <= 0:
            return (
                (self.net_premium) if self.net_premium else np.nan
            )  # Do not allow negative net premium. This is a safety net.
        return net_premium

    def calculate_bid_ask(self) -> Tuple[float, float]:
        """
        Calculate the bid-ask spread for the entire option strategy.

        Returns:
            Tuple[float, float]: A tuple containing (bid, ask) for the strategy.
            For a credit spread, both values will be positive (you receive a credit).
            For a debit spread, both values will be negative (you pay a debit).
        """
        strategy_bid = 0
        strategy_ask = 0

        for leg, ratio in zip(self.legs, self.leg_ratios):
            if leg.current_bid is None or leg.current_ask is None:
                continue

            if leg.position_side == "BUY":
                # When buying, we pay the ask and receive the bid
                strategy_bid -= leg.current_ask * ratio  # Cost (negative)
                strategy_ask -= leg.current_bid * ratio  # Cost (negative)
            elif leg.position_side == "SELL":
                # When selling, we receive the bid and pay the ask
                strategy_bid += leg.current_bid * ratio  # Credit (positive)
                strategy_ask += leg.current_ask * ratio  # Credit (positive)

        strategy_bid = abs(strategy_bid)
        strategy_ask = abs(strategy_ask)
        strategy_bid, strategy_ask = sorted([strategy_bid, strategy_ask])

        # Cap bid and ask to max_exit_net_premium if it exists
        if hasattr(self, "max_exit_net_premium"):
            strategy_bid = min(strategy_bid, self.max_exit_net_premium)
            strategy_ask = min(strategy_ask, self.max_exit_net_premium)

        return strategy_bid, strategy_ask

    def set_attribute(self, attr_name, attr_value):
        """
        Set an attribute dynamically.

        Args:
            attr_name (str): The name of the attribute to set.
            attr_value: The value to set the attribute to.

        Note:
            This method provides a way to dynamically set attributes on the strategy object,
            which can be useful for custom exit conditions or strategy modifications.
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, attr_value)
        else:
            raise AttributeError(
                f"'OptionStrategy' object has no attribute '{attr_name}'"
            )

    def __repr__(self):
        """
        Return a string representation of the OptionStrategy object.

        Returns:
            str: A string containing the strategy type, symbol, and status.
        """
        legs_repr = "\n    ".join(repr(leg) for leg in self.legs)
        return (
            f"OptionStrategy(\n"
            f"  Symbol: {self.symbol},\n"
            f"  Strategy Type: {self.strategy_type},\n"
            f"  Status: {self.status},\n"
            f"  Contracts: {self.contracts},\n"
            f"  Entry Time: {self.entry_time},\n"
            f"  Exit Time: {self.exit_time},\n"
            f"  Current Time: {self.current_time},\n"
            f"  Profit Target: {self.profit_target},\n"
            f"  Stop Loss: {self.stop_loss},\n"
            f"  Trailing Stop: {self.trailing_stop},\n"
            f"  Highest Return: {self.highest_return:.2f}%,\n"
            f"  Entry Net Premium: {self.entry_net_premium:.2f},\n"
            f"  Exit Net Premium: {self.exit_net_premium},\n"
            f"  Net Premium: {self.net_premium:.2f},\n"
            f"  Return Percentage: {self.return_percentage():.2f}%,\n"
            f"  Current Delta: {self.current_delta():.2f},\n"
            f"  Days in Trade (DIT): {self.DIT},\n"
            f"  Entry Rate of Return (ROR): {self.entry_ror},\n"
            f"  Exit Rate of Return (ROR): {self.exit_ror},\n"
            f"  Entry Underlying Last: {self.entry_underlying_last},\n"
            f"  Exit Underlying Last: {self.exit_underlying_last},\n"
            f"  Exit DIT: {self.exit_dit},\n"
            f"  Exit DTE: {self.exit_dte},\n"
            f"  Strategy Bid: {self.current_bid:.2f},\n"
            f"  Strategy Ask: {self.current_ask:.2f},\n"
            f"  Legs:\n    {legs_repr}\n"
            f"  Total Commission: {self.calculate_total_commission():.2f},\n"
            f"  Exit Scheme:{self.exit_scheme.__repr__() if hasattr(self, 'exit_scheme') and self.exit_scheme is not None else None}\n"
            f")"
        )

    def return_over_risk(self) -> float:
        """
        Calculate the current return over risk value for the spread.
        This represents the current potential return divided by the current potential risk.

        Returns:
            float: The current return over risk ratio, or infinity if the risk is zero.
        """

        if self.strategy_type in ["Vertical Spread", "Iron Condor", "Butterfly"]:
            max_risk = self.get_required_capital_per_contract()
            return_over_risk = (
                self.net_premium * 100 / max_risk if max_risk != 0 else float("inf")
            )
        elif self.strategy_type in ["Straddle", "Naked Call", "Naked Put"]:
            # For strategies with theoretically unlimited risk
            return_over_risk = float("inf")
        else:
            return_over_risk = float("inf")  # Default to infinity if not applicable

        return return_over_risk
