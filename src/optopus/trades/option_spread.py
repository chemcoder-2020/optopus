import pandas as pd
from pandas import Timestamp, Timedelta
from .option_leg import OptionLeg
import datetime
import numpy as np
import time  # Added for performance profiling
from loguru import logger
import cProfile
import pstats
from pstats import SortKey
from .exit_conditions import DefaultExitCondition, ExitConditionChecker
from .option_chain_converter import OptionChainConverter


class OptionStrategy:
    """
    Represents an options trading strategy composed of multiple option legs.

    Attributes:
        symbol (str): The underlying asset symbol.
        strategy_type (str): The type of option strategy.
        legs (list): List of OptionLeg objects comprising the strategy.
        entry_time (datetime): The entry time for the strategy.
        current_time (datetime): The current time.
        status (str): The status of the strategy ("OPEN" or "CLOSED").
        profit_target (float): Profit target percentage.
        stop_loss (float): Stop loss percentage.
        trailing_stop (float): Trailing stop percentage.
        highest_return (float): Highest return percentage achieved.
        entry_net_premium (float): Net premium at entry.
        net_premium (float): Current net premium.
        won (bool or None): Whether the trade was won (True), lost (False), or is still open (None).
        DIT (int): Days in Trade, representing the number of calendar days since the trade was opened.
        contracts (int): The number of contracts for the strategy.
        commission (float): The commission per contract.

    Methods:
        add_leg(leg): Add an option leg to the strategy.
        update(current_time, option_chain_df): Update the strategy with new market data.
        close_strategy(close_time, option_chain_df): Close the option strategy.
        total_pl(): Calculate the total profit/loss of the strategy.
        return_percentage(): Calculate the return percentage of the strategy.
        current_delta(): Calculate the current delta of the strategy.
    """

    def __init__(
        self,
        symbol: str,
        strategy_type: str,
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        contracts: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
    ):
        """
        Initialize an OptionStrategy object.

        Args:
            symbol (str): The underlying asset symbol.
            strategy_type (str): The type of option strategy.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            contracts (int, optional): The number of contracts for the strategy. Defaults to 1.
            exit_scheme (ExitConditionChecker, optional): The exit condition scheme to use. Defaults to DefaultExitCondition.
        """
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
        """Convert time to a standard pandas Timestamp object."""
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
        return self._contracts

    @contracts.setter
    def contracts(self, value):
        self._contracts = value
        self._update_leg_contracts()

    def _update_leg_contracts(self):
        for leg, ratio in zip(self.legs, self.leg_ratios):
            leg.contracts = self._contracts * ratio

    def add_leg(self, leg: OptionLeg, ratio: int = 1):
        """
        Add an option leg to the strategy.

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

        # Adjust premium based on position side
        # premium_adjustment = leg.entry_price * ratio
        # if leg.position_side == "SELL":
        #     self.entry_net_premium += premium_adjustment
        # else:  # BUY
        #     self.entry_net_premium -= premium_adjustment

        # self.net_premium = self.entry_net_premium

    def update(self, current_time: str, option_chain_df: pd.DataFrame):
        """
        Update the strategy with new market data and check exit conditions. This method will close the strategy if the exit conditions are met.

        Args:
            current_time (str): The current time for evaluation.
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

        self.net_premium = self.calculate_net_premium()

        if self.status == "OPEN":
            self._check_exit_conditions(option_chain_df)

        # Update exit_dte
        if self.status == "OPEN":
            self.exit_dte = (
                pd.to_datetime(self.legs[0].expiration) - self.current_time
            ).days

        # Calculate and store the strategy's bid-ask spread
        self.current_bid, self.current_ask = self.calculate_bid_ask()

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

        # Perform any final calculations or updates here
        for leg in self.legs:
            leg.update(self.current_time, option_chain_df)

        self.status = "CLOSED"
        self.won = self.total_pl() > 0
        self.exit_time = self.current_time
        self.exit_net_premium = self.net_premium
        self.exit_ror = self.return_over_risk()
        self.exit_underlying_last = self.legs[0].underlying_last
        self.exit_dit = self.DIT
        self.exit_dte = (pd.to_datetime(self.legs[0].expiration) - self.exit_time).days

    def _reopen_strategy(self):
        self.status = "OPEN"
        self.won = None
        self.exit_time = None
        self.exit_net_premium = None
        self.exit_ror = None
        self.exit_underlying_last = None
        self.exit_dit = None
        self.exit_dte = None
        self.highest_return = max(0, self.return_percentage())

    def close_strategy(self, close_time: str, option_chain_df: pd.DataFrame):
        """
        Close the option strategy.

        Args:
            close_time (str): The time at which the strategy is being closed.
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

    def total_pl(self):
        """Calculate the total profit/loss of the strategy."""
        # return sum(leg.calculate_pl() for leg in self.legs)
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

    def return_percentage(self):
        """Calculate the return percentage of the strategy."""
        premium = abs(self.entry_net_premium)
        if premium == 0:
            return 0
        return (self.total_pl() / (premium * 100 * self.contracts)) * 100

    def current_delta(self):
        """
        Calculate the current delta of the strategy.
        Takes into account position side (BUY/SELL) and contract multiplier.
        """
        return sum(
            leg.current_delta
            * leg.contracts
            * 100
            * (1 if leg.position_side == "BUY" else -1)
            for leg in self.legs
        )

    def conflicts_with(self, other_spread):
        """
        Check if this option spread conflicts with another option spread.

        Args:
            other_spread (OptionSpread): The other option spread to compare with.

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
    def get_strike_value(cls, converter, strike_input, expiration_date, option_type, reference_strike=None):
        if isinstance(strike_input, (int, float)):
            if abs(strike_input) < 1:
                # Numeric input treated as delta if float < 1
                return converter.get_desired_strike(
                    expiration_date,
                    option_type,
                    strike_input,
                    by="delta"
                )
            else:
                # Numeric input treated as a specific strike price
                return converter.get_desired_strike(
                    expiration_date,
                    option_type,
                    strike_input,
                    by="strike"
                )
        elif isinstance(strike_input, str):
            if strike_input.startswith(("+", "-")):
                try:
                    offset = float(strike_input[1:])
                    if abs(offset) < 1:
                        # ATM relative strike with delta
                        return converter.get_desired_strike(
                            expiration_date,
                            option_type,
                            offset,
                            by="delta"
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
                        expiration_date,
                        option_type,
                        strike_price,
                        by="strike"
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
    def create_vertical_spread(
        cls,
        symbol: str,
        option_type: str,
        long_strike,
        short_strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
    ):
        """
        Create a vertical spread option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            option_type (str): The option type ('CALL' or 'PUT').
            long_strike: The strike price, delta, or ATM offset (e.g., "+2", 0.3, or "ATM") for long leg.
            short_strike: The strike price, delta, or ATM offset for short leg.
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.
            leg_ratio (int, optional): The ratio of leg contracts to the strategy's contract count.
            commission (float, optional): The commission per contract per leg.

        Returns:
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
            converter, long_strike, expiration_date, option_type, reference_strike=short_strike_value if isinstance(long_strike, str) and (long_strike[0] == "+" or long_strike[0] == "-") else None
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

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        if strategy.entry_net_premium > abs(short_strike_value - long_strike_value):
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
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
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
            commission (float, optional): The commission per contract per leg.

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
            converter, put_long_strike, expiration_date, "PUT", reference_strike=put_short_strike_value if isinstance(put_long_strike, str) and (put_long_strike[0] == "+" or put_long_strike[0] == "-") else None
        )

        # Get call strikes
        call_short_strike_value = strategy.get_strike_value(
            converter, call_short_strike, expiration_date, "CALL"
        )
        call_long_strike_value = strategy.get_strike_value(
            converter, call_long_strike, expiration_date, "CALL", reference_strike=call_short_strike_value if isinstance(call_long_strike, str) and (call_long_strike[0] == "+" or call_long_strike[0] == "-") else None
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

        strategy.entry_time = cls._standardize_time(entry_time)
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
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        leg_ratio: int = 1,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
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
            commission (float, optional): The commission per contract per leg.

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

        # If strike is "ATM", use the ATM strike, otherwise use the specified strike
        if strike == "ATM":
            strike_value = converter.get_atm_strike(expiration_date)
        else:
            strike_value = converter.get_desired_strike(
                expiration_date,
                "CALL",
                strike,
                by="delta" if isinstance(strike, float) else "strike",
            )

        call_leg = OptionLeg(
            symbol,
            "CALL",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )
        put_leg = OptionLeg(
            symbol,
            "PUT",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )

        strategy.strategy_side = "DEBIT"

        strategy.add_leg(call_leg, leg_ratio)
        strategy.add_leg(put_leg, leg_ratio)

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    @classmethod
    def create_butterfly(
        cls,
        symbol: str,
        option_type: str,
        lower_strike,
        middle_strike,
        upper_strike,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
    ):
        """
        Create a butterfly option strategy.

        Args:
            symbol (str): The underlying asset symbol.
            option_type (str): The option type, either "CALL" or "PUT".
            lower_strike: The lower strike price, delta, or ATM offset (e.g., "-2", -0.3, or "ATM").
            middle_strike: The middle strike price, delta, or ATM offset.
            upper_strike: The upper strike price, delta, or ATM offset (e.g., "+2", 0.3, or "ATM").
            expiration (str or int): The option expiration date or target DTE.
            contracts (int): The number of contracts for the strategy (will be doubled for the middle leg).
            entry_time (str): The entry time for the strategy.
            option_chain_df (pd.DataFrame): The option chain data.
            profit_target (float, optional): Profit target percentage.
            stop_loss (float, optional): Stop loss percentage.
            trailing_stop (float, optional): Trailing stop percentage.

        Returns:
            OptionStrategy: A butterfly strategy object.
        """
        converter = OptionChainConverter(option_chain_df)

        strategy = cls(
            symbol,
            "Butterfly",
            profit_target,
            stop_loss,
            trailing_stop,
            contracts,
            commission,
            exit_scheme,
        )

        expiration_date = converter.get_closest_expiration(expiration)

        # Determine strike selection method and get strikes

        lower_strike_value = strategy.get_strike_value(lower_strike)
        middle_strike_value = strategy.get_strike_value(middle_strike)
        upper_strike_value = strategy.get_strike_value(upper_strike)

        lower_leg = OptionLeg(
            symbol,
            option_type,
            lower_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )
        middle_leg = OptionLeg(
            symbol,
            option_type,
            middle_strike_value,
            expiration_date,
            contracts * 2,
            entry_time,
            option_chain_df,
            "SELL",
            commission=commission,
        )
        upper_leg = OptionLeg(
            symbol,
            option_type,
            upper_strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )

        strategy.strategy_side = "CREDIT"

        strategy.add_leg(lower_leg, 1)
        strategy.add_leg(middle_leg, 2)
        strategy.add_leg(upper_leg, 1)

        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.entry_ror = strategy.return_over_risk()
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

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
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
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

        # Determine strike selection method
        if isinstance(strike, (int, float)):
            # Numeric input treated as delta if float < 1, otherwise as strike price
            strike_value = converter.get_desired_strike(
                expiration_date,
                "CALL",
                strike,
                by="delta" if abs(float(strike)) < 1 else "strike",
            )
        elif isinstance(strike, str):
            if strike.upper() == "ATM":
                strike_value = converter.get_atm_strike(expiration_date)
            elif strike.startswith(("+", "-")):
                # ATM relative strike
                offset = float(strike)
                strike_value = converter.get_desired_strike(
                    expiration_date, "CALL", offset, by="atm"
                )
            else:
                # Try to convert to float for direct strike price
                try:
                    strike_price = float(strike)
                    strike_value = converter.get_desired_strike(
                        expiration_date, "CALL", strike_price, by="strike"
                    )
                except ValueError:
                    raise ValueError(f"Invalid strike input: {strike}")
        else:
            raise ValueError(f"Unsupported strike input type: {type(strike)}")

        call_leg = OptionLeg(
            symbol,
            "CALL",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )

        strategy.strategy_side = "DEBIT"

        strategy.add_leg(call_leg)
        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
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
        profit_target: float = None,
        stop_loss: float = None,
        trailing_stop: float = None,
        commission: float = 0.5,
        exit_scheme: ExitConditionChecker = None,
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

        # Determine strike selection method
        if isinstance(strike, (int, float)):
            # Numeric input treated as delta if float < 1, otherwise as strike price
            strike_value = converter.get_desired_strike(
                expiration_date,
                "PUT",
                strike,
                by="delta" if abs(float(strike)) < 1 else "strike",
            )
        elif isinstance(strike, str):
            if strike.upper() == "ATM":
                strike_value = converter.get_atm_strike(expiration_date)
            elif strike.startswith(("+", "-")):
                # ATM relative strike
                offset = float(strike)
                strike_value = converter.get_desired_strike(
                    expiration_date, "PUT", offset, by="atm"
                )
            else:
                # Try to convert to float for direct strike price
                try:
                    strike_price = float(strike)
                    strike_value = converter.get_desired_strike(
                        expiration_date, "PUT", strike_price, by="strike"
                    )
                except ValueError:
                    raise ValueError(f"Invalid strike input: {strike}")
        else:
            raise ValueError(f"Unsupported strike input type: {type(strike)}")

        put_leg = OptionLeg(
            symbol,
            "PUT",
            strike_value,
            expiration_date,
            contracts,
            entry_time,
            option_chain_df,
            "BUY",
            commission=commission,
        )

        strategy.strategy_side = "DEBIT"

        strategy.add_leg(put_leg)
        strategy.entry_net_premium = strategy.net_premium = (
            strategy.calculate_net_premium()
        )

        strategy.entry_time = cls._standardize_time(entry_time)
        strategy.current_bid, strategy.current_ask = strategy.calculate_bid_ask()

        return strategy

    def get_required_capital(self) -> float:
        """
        Calculate the required capital for the option strategy.
        This is typically the maximum potential loss of the strategy.

        Returns:
            float: The required capital for the strategy.
        """

        if self.strategy_type in ["Vertical Spread", "Iron Condor"]:
            max_width = max(
                abs(leg1.strike - leg2.strike)
                for leg1, leg2 in zip(self.legs[::2], self.legs[1::2])
            )
            required_capital = (
                (max_width - self.entry_net_premium) * 100 * self.contracts
            )
        elif self.strategy_type == "Straddle":
            required_capital = self.entry_net_premium * 100 * self.contracts
        elif self.strategy_type == "Butterfly":
            wing_width = self.legs[1].strike - self.legs[0].strike
            required_capital = wing_width * 100 * self.contracts
        elif self.strategy_type in ["Naked Call", "Naked Put"]:
            required_capital = abs(self.entry_net_premium * 100 * self.contracts)
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        return required_capital + self.calculate_total_commission()

    def get_required_capital_per_contract(self) -> float:
        """
        Calculate the required capital per contract for the option strategy.
        This is typically the maximum potential loss of the strategy per contract.

        Returns:
            float: The required capital per contract for the strategy.
        """

        required_capital_per_contract = self.get_required_capital() / self.contracts

        return required_capital_per_contract

    def calculate_net_premium(self) -> float:
        """
        Calculate the net premium based on the current prices and position sides of the legs.

        Returns:
            float: The calculated net premium.
        """
        bid, ask = self.calculate_bid_ask()
        net_premium = (bid + ask) / 2
        if net_premium <= 0:
            return (
                (self.net_premium) if self.net_premium else np.nan
            )  # Do not allow negative net premium. This is a safety net.
        return net_premium

    def calculate_bid_ask(self):
        """
        Calculate the bid-ask spread for the entire option strategy.

        Returns:
            tuple: A tuple containing (bid, ask) for the strategy.
        """
        strategy_bid = 0
        strategy_ask = 0

        for leg, ratio in zip(self.legs, self.leg_ratios):
            if leg.position_side == "BUY":
                if leg.current_ask is not None:
                    strategy_ask += leg.current_ask * ratio
                if leg.current_bid is not None:
                    strategy_bid += leg.current_bid * ratio
            elif leg.position_side == "SELL":
                if leg.current_bid is not None:
                    strategy_ask -= leg.current_bid * ratio
                if leg.current_ask is not None:
                    strategy_bid -= leg.current_ask * ratio

        strategy_bid = abs(strategy_bid)
        strategy_ask = abs(strategy_ask)
        return min(strategy_bid, strategy_ask), max(strategy_bid, strategy_ask)

    def set_attribute(self, attr_name, attr_value):
        """
        Set an attribute dynamically.

        Args:
            attr_name (str): The name of the attribute to set.
            attr_value: The value to set the attribute to.
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, attr_value)
        else:
            raise AttributeError(
                f"'OptionStrategy' object has no attribute '{attr_name}'"
            )

    def __repr__(self):
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

    def return_over_risk(self):
        """
        Calculate the current return over risk value for the spread.
        This represents the current potential return divided by the current potential risk.

        Returns:
            float: The current return over risk ratio.
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


if __name__ == "__main__":
    # Load the entry option chain data
    entry_file = "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar/SPY_2024-09-06 15-30.parquet"
    entry_df = pd.read_parquet(entry_file)

    # Load the update option chain data
    update_file = "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar/SPY_2024-09-06 15-45.parquet"
    update_df = pd.read_parquet(update_file)

    update_file2 = "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-09 09-45.parquet"
    update_df2 = pd.read_parquet(update_file2)

    print("\nRunning tests:")

    print("\n--- Vertical Spread Tests ---")

    # Create a vertical spread
    # profiler = cProfile.Profile()
    # profiler.enable()
    vertical_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",  # This will be calculated first
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    # stats.print_stats()

    print("Vertical Spread created:", vertical_spread)

    # Test 1: Updating with the same time and df should result in identical spread
    vertical_spread_copy = vertical_spread
    vertical_spread_copy.update("2024-09-06 15:30:00", entry_df)

    assert (
        vertical_spread_copy.total_pl() == 0
    ), f"Test 1 failed: P&L {vertical_spread_copy.total_pl()} should be 0 when updating with the same time and df"
    print(
        f"Test 1 passed: Updating with the same time and df results in identical spread {vertical_spread_copy.total_pl()}"
    )

    # Test 2: Decreasing underlying price should result in negative P&L for call credit spread
    vertical_spread.update(
        "2024-09-06 15:45:00", update_df
    )  # update_df has lower underlying price

    assert (
        vertical_spread.total_pl() > 0
    ), f"Test 2 failed: P&L {vertical_spread.total_pl()} should be positive when underlying price decreases for call credit spread"
    print(
        f"Test 2 passed: Decreasing underlying price results in positive P&L {vertical_spread.total_pl()} for call credit spread"
    )

    # Test 3: Increasing underlying price should result in negative P&L for call credit spread
    vertical_spread.update(
        "2024-09-09 09:45:00", update_df2
    )  # update_df2 has higher underlying price

    assert (
        vertical_spread.total_pl() < 0
    ), f"Test 3 failed: P&L {vertical_spread.total_pl()} should be negative when underlying price increases for call credit spread"
    print(
        f"Test 3 passed: Increasing underlying price results in negative P&L {vertical_spread.total_pl()} for call credit spread"
    )
    # Test 4: Total cost should remain constant after updates
    entry_net_premium = vertical_spread.entry_net_premium

    vertical_spread.update("2024-09-06 15:45:00", update_df)
    new_premium = vertical_spread.entry_net_premium

    assert (
        abs(entry_net_premium - new_premium) < 0.01
    ), "Test 4 failed: Entry net premium should remain constant after updates"
    print("Test 4 passed: Entry net premium remains constant after updates")

    # Test 5: ATM strike selection
    atm_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="ATM",
        short_strike="ATM",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(
        f"Test 5: ATM spread created - Long strike: {atm_spread.legs[0].strike}, Short strike: {atm_spread.legs[1].strike}"
    )
    assert (
        atm_spread.legs[0].strike == atm_spread.legs[1].strike
    ), "ATM strikes should be equal"
    print("Test 5 passed: ATM strikes are equal")

    # Test 6: Specific delta strike selection
    delta_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike="-0.15",
        short_strike="-0.3",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(
        f"Test 6: Delta spread created - Long strike: {delta_spread.legs[0].strike}, Short strike: {delta_spread.legs[1].strike}"
    )
    assert (
        delta_spread.legs[0].strike < delta_spread.legs[1].strike
    ), "Long strike should be lower than short strike for put spread"
    print("Test 6 passed: Delta strikes are correctly ordered")

    # Test 7: Relative strike selection
    relative_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+5",
        short_strike="ATM",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(
        f"Test 7: Relative spread created - Long strike: {relative_spread.legs[0].strike}, Short strike: {relative_spread.legs[1].strike}"
    )
    assert (
        relative_spread.legs[0].strike > relative_spread.legs[1].strike
    ), "Long strike should be higher than short strike for call spread"
    assert (
        abs(relative_spread.legs[0].strike - relative_spread.legs[1].strike) == 5
    ), "Strike difference should be 5"
    print("Test 7 passed: Relative strikes are correctly calculated")

    # Test 8: Returns over Risk calculation for vertical spread
    entry_ror = vertical_spread.return_over_risk()
    current_ror = vertical_spread.return_over_risk()
    print(f"Test 8: Vertical Spread Entry Returns over Risk: {entry_ror:.4f}")
    print(f"Test 8: Vertical Spread Current Returns over Risk: {current_ror:.4f}")
    assert (
        entry_ror > 0
    ), "Entry Returns over Risk should be positive for a credit spread"
    assert (
        current_ror > 0
    ), "Current Returns over Risk should be positive for a credit spread"
    print(
        "Test 8 passed: Returns over Risk calculated successfully for vertical spread"
    )

    print("\n--- Iron Condor Tests ---")

    # Create an iron condor
    iron_condor = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike="-5",
        put_short_strike="-0.3",  # This will be calculated first for puts
        call_short_strike="+0.3",  # This will be calculated first for calls
        call_long_strike="+5",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print("Iron Condor created:", iron_condor)

    # Test 9: Iron Condor creation and strike ordering
    print(
        f"Test 9: Iron Condor created - Put long: {iron_condor.legs[0].strike}, Put short: {iron_condor.legs[1].strike}, Call short: {iron_condor.legs[2].strike}, Call long: {iron_condor.legs[3].strike}"
    )
    assert (
        iron_condor.legs[0].strike
        < iron_condor.legs[1].strike
        < iron_condor.legs[2].strike
        < iron_condor.legs[3].strike
    ), "Strikes should be in ascending order"
    print("Test 9 passed: Iron Condor strikes are correctly ordered")

    # Test 10: Iron Condor P&L calculation
    iron_condor.update("2024-09-06 15:45:00", update_df)
    print(f"Test 10: Iron Condor P&L after update: ${iron_condor.total_pl():.2f}")
    assert isinstance(iron_condor.total_pl(), float), "P&L should be a float value"
    print("Test 10 passed: Iron Condor P&L calculated successfully")

    # Test 11: Iron Condor Returns over Risk calculation
    ic_ror = iron_condor.return_over_risk()
    print(f"Test 11: Iron Condor Returns over Risk: {ic_ror:.4f}")
    assert ic_ror > 0, "Returns over Risk should be positive for an iron condor"
    print("Test 11 passed: Returns over Risk calculated successfully for iron condor")

    # Test 12: Iron Condor delta calculation
    initial_delta = iron_condor.current_delta()
    iron_condor.update("2024-09-06 15:45:00", update_df)
    updated_delta = iron_condor.current_delta()
    print(
        f"Test 12: Iron Condor initial delta: {initial_delta:.2f}, updated delta: {updated_delta:.2f}"
    )
    assert (
        abs(initial_delta) < 10 and abs(updated_delta) < 10
    ), "Delta should remain close to zero for an iron condor"
    print("Test 12 passed: Iron Condor delta remains close to zero")

    # Test 13: Iron Condor delta change with significant price movement
    iron_condor.update(
        "2024-09-09 09:45:00", update_df2
    )  # Assuming this represents a larger price move
    significant_move_delta = iron_condor.current_delta()
    print(
        f"Test 13: Iron Condor delta after significant move: {significant_move_delta:.2f}"
    )
    assert abs(significant_move_delta) > abs(
        initial_delta
    ), "Delta should change more with a significant price movement"
    print("Test 13 passed: Iron Condor delta changes with significant price movement")

    print("\n--- Straddle Tests ---")

    # Create a straddle
    straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike="ATM",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print("Straddle created:", straddle)

    # Test 13: Straddle creation and strike equality
    print(
        f"Test 13: Straddle created - Call strike: {straddle.legs[0].strike}, Put strike: {straddle.legs[1].strike}"
    )
    assert (
        straddle.legs[0].strike == straddle.legs[1].strike
    ), "Call and Put strikes should be equal in a straddle"
    print("Test 13 passed: Straddle strikes are equal")

    # Test 14: Straddle P&L calculation
    straddle.update("2024-09-06 15:45:00", update_df)
    print(f"Test 14: Straddle P&L after update: ${straddle.total_pl():.2f}")
    assert isinstance(straddle.total_pl(), float), "P&L should be a float value"
    print("Test 14 passed: Straddle P&L calculated successfully")

    # Test 15: Straddle Returns over Risk calculation
    straddle_ror = straddle.return_over_risk()
    print(f"Test 15: Straddle Returns over Risk: {straddle_ror:.4f}")
    assert straddle_ror == float(
        "inf"
    ), "Returns over Risk should be infinite for a straddle"
    print("Test 15 passed: Returns over Risk calculated successfully for straddle")

    # Test 16: Straddle delta calculation
    straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike="ATM",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    initial_straddle_delta = straddle.current_delta()
    straddle.update("2024-09-06 15:45:00", update_df)
    updated_straddle_delta = straddle.current_delta()
    print(
        f"Test 16: Straddle initial delta: {initial_straddle_delta:.2f}, updated delta: {updated_straddle_delta:.2f}"
    )
    assert (
        initial_straddle_delta != updated_straddle_delta
    ), "Delta should change after update"
    print("Test 16 passed: Straddle delta calculated and updated successfully")

    print("\n--- Additional Vertical Spread Tests ---")

    # Create a put vertical spread
    put_vertical = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike="-5",
        short_strike="-0.3",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print("Put Vertical Spread created:", put_vertical)

    # Test 17: Put Vertical Spread Returns over Risk calculation
    put_ror = put_vertical.return_over_risk()
    print(f"Test 17: Put Vertical Spread Returns over Risk: {put_ror:.4f}")
    assert put_ror > 0, "Returns over Risk should be positive for a put credit spread"
    print(
        "Test 17 passed: Returns over Risk calculated successfully for put vertical spread"
    )

    # Test 18: Put Vertical Spread delta calculation
    initial_put_delta = put_vertical.current_delta()
    put_vertical.update("2024-09-06 15:45:00", update_df)
    updated_put_delta = put_vertical.current_delta()
    print(
        f"Test 18: Put Vertical initial delta: {initial_put_delta:.2f}, updated delta: {updated_put_delta:.2f}"
    )
    assert (
        initial_put_delta > updated_put_delta
    ), "Delta should decrease for put spread when underlying price decreases"
    print("Test 18 passed: Put Vertical delta calculated and updated successfully")

    # Additional tests
    print("\n--- Test OptionStrategy update with different current_time formats ---")
    strategy = OptionStrategy("SPY", "Vertical Spread")
    strategy.add_leg(
        OptionLeg(
            symbol="SPY",
            option_type="PUT",
            strike=541.0,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=entry_df,
            position_side="SELL",
        )
    )
    strategy.update("2024-09-06 15:45:00", update_df)
    print("Strategy after update:", strategy)

    strategy.update(pd.Timestamp("2024-09-06 15:45:00"), update_df)
    print("Strategy after update:", strategy)

    strategy.update(datetime.datetime(2024, 9, 6, 15, 45, 0), update_df)
    print("Strategy after update:", strategy)

    # Test 19: Closing the strategy and checking the 'won' attribute
    vertical_spread.close_strategy("2024-09-09 09:45:00", update_df2)
    print(f"Test 19: Vertical Spread status after closing: {vertical_spread.status}")
    print(f"Test 19: Vertical Spread 'won' attribute: {vertical_spread.won}")
    assert vertical_spread.status == "CLOSED", "Status should be CLOSED after closing"

    assert isinstance(vertical_spread.won, np.bool_), "'won' should be a boolean value"
    print("Test 19 passed: Strategy closed and 'won' attribute set correctly")

    # Test 20: Closing an already closed strategy should raise ValueError
    try:
        vertical_spread.close_strategy("2024-09-09 10:00:00", update_df2)
        assert (
            False
        ), "ValueError should be raised when closing an already closed strategy"
    except ValueError:
        print(
            "Test 20 passed: ValueError raised when closing an already closed strategy"
        )

    # Test 21: Closing a strategy with positive P&L should set 'won' to True
    positive_pl_strategy = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    positive_pl_strategy.update(
        "2024-09-06 15:45:00", update_df
    )  # Assuming this results in positive P&L
    positive_pl_strategy.close_strategy("2024-09-06 15:45:00", update_df)
    print(f"Test 21: Positive P&L strategy 'won' attribute: {positive_pl_strategy.won}")
    assert positive_pl_strategy.won == True, "'won' should be True for positive P&L"
    print("Test 21 passed: 'won' set to True for positive P&L")

    # Test 22: Closing a strategy with negative P&L should set 'won' to False
    negative_pl_strategy = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",
        expiration="2024-10-31",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    negative_pl_strategy.update(
        "2024-09-09 09:45:00", update_df2
    )  # Assuming this results in negative P&L
    negative_pl_strategy.close_strategy("2024-09-09 09:45:00", update_df2)
    print(f"Test 22: Negative P&L strategy 'won' attribute: {negative_pl_strategy.won}")
    assert negative_pl_strategy.won == False, "'won' should be False for negative P&L"
    print("Test 22 passed: 'won' set to False for negative P&L")

    print("Test 23: OptionSpread Conflict Detection Test")

    # Test: OptionSpread conflict detection
    spread1 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    spread2 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    spread3 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=540,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    assert spread1.conflicts_with(
        spread2
    ), "Test failed: Overlapping spreads should conflict"
    assert not spread1.conflicts_with(
        spread3
    ), "Test failed: Different option type spreads should not conflict"
    print("OptionSpread conflict detection test passed")

    print("\nConflict Detection Tests:")

    # Test 23: Basic Vertical Spread Conflict
    print("Test 23: Basic Vertical Spread Conflict")
    spread1 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    spread2 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert spread1.conflicts_with(spread2), "Identical spreads should conflict"
    print("Test 23 passed: Identical spreads conflict")

    # Test 24: Different Expiration
    print("Test 24: Different Expiration")
    spread3 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2025-01-17",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert not spread1.conflicts_with(
        spread3
    ), "Spreads with different expirations should not conflict"
    print("Test 24 passed: Different expirations don't conflict")

    # Test 25: Different Entry Time
    print("Test 25: Different Entry Time")
    entry_df2 = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-09 10-00.parquet"
    )
    spread4 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-09 10:00:00",
        option_chain_df=entry_df2,
    )
    assert spread1.conflicts_with(
        spread4
    ), "Spreads with different entry times but same legs should conflict"
    print("Test 25 passed: Different entry times still conflict")

    # Test 26: Different Number of Contracts
    print("Test 26: Different Number of Contracts")
    spread5 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=2,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert spread1.conflicts_with(
        spread5
    ), "Spreads with different contract numbers but same legs should conflict"
    print("Test 26 passed: Different contract numbers still conflict")

    # Test 27: Similar but Non-Conflicting Legs
    print("Test 27: Similar but Non-Conflicting Legs")
    spread6 = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=570,
        short_strike=560,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert spread1.conflicts_with(
        spread6
    ), "Spreads 1 conflicts with spread 6 on short strike at 560"
    print("Test 27 passed: shared legs conflict")

    # Test 28: Iron Condor Conflict
    print("Test 28: Iron Condor Conflict")
    iron_condor1 = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike=540,
        put_short_strike=550,
        call_short_strike=570,
        call_long_strike=580,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    iron_condor2 = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike=540,
        put_short_strike=550,
        call_short_strike=570,
        call_long_strike=580,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert iron_condor1.conflicts_with(
        iron_condor2
    ), "Identical iron condors should conflict"
    print("Test 28 passed: Identical iron condors conflict")

    # Test 29: Iron Condor and Vertical Spread Conflict
    print("Test 29: Iron Condor and Vertical Spread Conflict")
    assert not iron_condor1.conflicts_with(
        spread1
    ), "Iron condor should not conflict with vertical spread on same strike because it's a different option type"
    print(
        "Test 29 passed: Iron condor should not conflict with vertical spread on same strike because it's a different option type"
    )

    # Test 30: Straddle Conflict
    print("Test 30: Straddle Conflict")
    straddle1 = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    straddle2 = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert straddle1.conflicts_with(straddle2), "Identical straddles should conflict"
    print("Test 30 passed: Identical straddles conflict")

    # Test 31: Straddle and Vertical Spread Conflict
    print("Test 31: Straddle and Vertical Spread Conflict")
    assert straddle1.conflicts_with(
        spread1
    ), "Straddle should conflict with overlapping vertical spread"
    print("Test 31 passed: Straddle conflicts with overlapping vertical spread")

    # Test 32: Non-Conflicting Strategies
    print("Test 32: Non-Conflicting Strategies")
    non_conflicting_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=530,
        short_strike=535,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert not spread1.conflicts_with(
        non_conflicting_spread
    ), "Different option types should not conflict"
    iron_condor1.legs
    assert not iron_condor1.conflicts_with(
        non_conflicting_spread
    ), "Non-overlapping iron condor and vertical spread should not conflict"
    assert not straddle1.conflicts_with(
        non_conflicting_spread
    ), "Non-overlapping straddle and vertical spread should not conflict"
    print("Test 32 passed: Non-conflicting strategies don't conflict")

    print("\nAll conflict detection tests completed!")
    print("\nTesting get_required_capital method:")

    # Test 33: Credit Call Spread
    print("Test 33: Credit Call Spread")
    credit_call_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    credit_call_capital = credit_call_spread.get_required_capital()
    expected_credit_call_capital = (
        (560 - 550 - credit_call_spread.entry_net_premium)
        * 100
        * credit_call_spread.contracts
    )
    print(f"Test 33: Credit Call Spread required capital: {credit_call_capital:.2f}")
    assert (
        credit_call_capital == expected_credit_call_capital
    ), "Credit call spread required capital calculation is incorrect"
    print("Test 33 passed: Credit call spread required capital calculated correctly")

    # Test 34: Credit Put Spread
    print("Test 34: Credit Put Spread")
    credit_put_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=540,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    credit_put_capital = credit_put_spread.get_required_capital()
    expected_credit_put_capital = (
        (550 - 540 - credit_put_spread.entry_net_premium)
        * 100
        * credit_put_spread.contracts
    )
    print(f"Test 34: Credit Put Spread required capital: {credit_put_capital:.2f}")
    assert (
        credit_put_capital == expected_credit_put_capital
    ), "Credit put spread required capital calculation is incorrect"
    print("Test 34 passed: Credit put spread required capital calculated correctly")

    # Test 35: Iron Condor
    print("Test 35: Iron Condor")
    iron_condor = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike=540,
        put_short_strike=550,
        call_short_strike=570,
        call_long_strike=580,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    iron_condor_capital = iron_condor.get_required_capital()
    expected_iron_condor_capital = (
        (max(abs(540 - 550), abs(570 - 580)) - iron_condor.entry_net_premium)
        * 100
        * iron_condor.contracts
    )

    print(f"Test 35: Iron Condor required capital: {iron_condor_capital:.2f}")
    assert (
        iron_condor_capital == expected_iron_condor_capital
    ), "Iron condor required capital calculation is incorrect"
    print("Test 35 passed: Iron condor required capital calculated correctly")

    # Test 36: Straddle
    print("Test 36: Straddle")
    straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    straddle_capital = straddle.get_required_capital()
    expected_straddle_capital = straddle.entry_net_premium * 100 * straddle.contracts
    print(f"Test 36: Straddle required capital: {straddle_capital:.2f}")
    assert (
        straddle_capital == expected_straddle_capital
    ), "Straddle required capital calculation is incorrect"
    print("Test 36 passed: Straddle required capital calculated correctly")

    print("\nTesting automatic strategy closure on update:")

    # Test 37: Strategy closure on profit target
    print("Test 37: Strategy closure on profit target")
    profit_target_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
        profit_target=1,  # 1% profit target
        stop_loss=10,  # 10% stop loss
    )

    # Simulate a profitable move
    profit_target_spread.update("2024-09-06 15:45:00", update_df)

    print(f"Strategy status after update: {profit_target_spread.status}")
    print(f"Strategy 'won' attribute: {profit_target_spread.won}")
    assert (
        profit_target_spread.status == "CLOSED"
    ), "Strategy should be closed after reaching profit target"
    assert profit_target_spread.won == True, "Strategy should be marked as won"
    print("Test 37 passed: Strategy closed automatically on reaching profit target")

    # Test 38: Strategy closure on stop loss
    print("Test 38: Strategy closure on stop loss")
    stop_loss_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
        profit_target=20,  # 20% profit target
        stop_loss=5,  # 5% stop loss
    )

    # Simulate a losing move
    stop_loss_spread.update("2024-09-09 9:45:00", update_df2)

    print(f"Strategy status after update: {stop_loss_spread.status}")
    print(f"Strategy 'won' attribute: {stop_loss_spread.won}")
    assert (
        stop_loss_spread.status == "CLOSED"
    ), "Strategy should be closed after hitting stop loss"
    assert stop_loss_spread.won == False, "Strategy should be marked as lost"
    print("Test 38 passed: Strategy closed automatically on hitting stop loss")

    print("\nTesting DIT (Days in Trade) functionality:")

    # Test 39: DIT calculation
    print("Test 39: DIT calculation")
    entry_df2 = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-03 15-30.parquet"
    )
    update_df3 = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-03 15-45.parquet"
    )
    dit_test_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-03 15:30:00",
        option_chain_df=entry_df2,
    )

    print(f"Initial DIT: {dit_test_spread.DIT}")
    assert dit_test_spread.DIT == 0, "Initial DIT should be 0"

    # Simulate updates on different days
    dit_test_spread.update("2024-09-03 15:45:00", update_df3)
    print(f"DIT after same day update: {dit_test_spread.DIT}")
    assert dit_test_spread.DIT == 0, "DIT should still be 0 on the same day"

    dit_test_spread.update(
        "2024-09-04 09:45:00",
        pd.read_parquet(
            "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-04 09-45.parquet"
        ),
    )
    print(f"DIT after next day update: {dit_test_spread.DIT}")
    assert dit_test_spread.DIT == 1, "DIT should be 1 on the next day"

    dit_test_spread.update(
        "2024-09-10 15:00:00",
        pd.read_parquet(
            "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-10 15-00.parquet"
        ),
    )
    print(f"DIT after one week: {dit_test_spread.DIT}")
    assert dit_test_spread.DIT == 7, "DIT should be 7 after one week"

    print("Test 39 passed: DIT calculation is correct")

    print("\nTesting get_required_capital_per_contract method:")

    # Test 40: Credit Call Spread (per contract)
    print("Test 40: Credit Call Spread (per contract)")
    credit_call_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    credit_call_capital_per_contract = (
        credit_call_spread.get_required_capital_per_contract()
    )
    expected_credit_call_capital_per_contract = (
        560 - 550 - credit_call_spread.entry_net_premium
    ) * 100
    print(
        f"Test 40: Credit Call Spread required capital per contract: {credit_call_capital_per_contract:.2f}"
    )
    assert (
        credit_call_capital_per_contract == expected_credit_call_capital_per_contract
    ), "Credit call spread required capital per contract calculation is incorrect"
    print(
        "Test 40 passed: Credit call spread required capital per contract calculated correctly"
    )

    # Test 41: Credit Put Spread (per contract)
    print("Test 41: Credit Put Spread (per contract)")
    credit_put_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=540,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    credit_put_capital_per_contract = (
        credit_put_spread.get_required_capital_per_contract()
    )
    expected_credit_put_capital_per_contract = (
        550 - 540 - credit_put_spread.entry_net_premium
    ) * 100
    print(
        f"Test 41: Credit Put Spread required capital per contract: {credit_put_capital_per_contract:.2f}"
    )
    assert (
        credit_put_capital_per_contract == expected_credit_put_capital_per_contract
    ), "Credit put spread required capital per contract calculation is incorrect"
    print(
        "Test 41 passed: Credit put spread required capital per contract calculated correctly"
    )

    # Test 42: Iron Condor (per contract)
    print("Test 42: Iron Condor (per contract)")
    iron_condor = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike=540,
        put_short_strike=550,
        call_short_strike=570,
        call_long_strike=580,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    iron_condor_capital_per_contract = iron_condor.get_required_capital_per_contract()
    expected_iron_condor_capital_per_contract = (
        max(abs(540 - 550), abs(570 - 580)) - iron_condor.entry_net_premium
    ) * 100

    print(
        f"Test 42: Iron Condor required capital per contract: {iron_condor_capital_per_contract:.2f}"
    )
    assert (
        iron_condor_capital_per_contract == expected_iron_condor_capital_per_contract
    ), "Iron condor required capital per contract calculation is incorrect"
    print(
        "Test 42 passed: Iron condor required capital per contract calculated correctly"
    )

    # Test 43: Straddle (per contract)
    print("Test 43: Straddle (per contract)")
    straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    straddle_capital_per_contract = straddle.get_required_capital_per_contract()
    expected_straddle_capital_per_contract = straddle.entry_net_premium * 100
    print(
        f"Test 43: Straddle required capital per contract: {straddle_capital_per_contract:.2f}"
    )
    assert (
        straddle_capital_per_contract == expected_straddle_capital_per_contract
    ), "Straddle required capital per contract calculation is incorrect"
    print("Test 43 passed: Straddle required capital per contract calculated correctly")

    print("\n--- DTE-based Expiration Selection Tests ---")

    # Test 44: Vertical Spread with DTE-based expiration
    print("Test 44: Vertical Spread with DTE-based expiration")
    dte_vertical_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",
        expiration=45,  # 45 DTE
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(
        f"DTE-based Vertical Spread expiration: {dte_vertical_spread.legs[0].expiration}"
    )
    assert (
        pd.to_datetime(dte_vertical_spread.legs[0].expiration)
        - pd.to_datetime("2024-09-06")
    ).days >= 45, "Expiration should be at least 45 days from entry"
    print("Test 44 passed: DTE-based Vertical Spread created successfully")

    # Test 45: Iron Condor with DTE-based expiration
    print("Test 45: Iron Condor with DTE-based expiration")
    dte_iron_condor = OptionStrategy.create_iron_condor(
        symbol="SPY",
        put_long_strike="-2",
        put_short_strike="-0.3",
        call_short_strike="+0.3",
        call_long_strike="+2",
        expiration=60,  # 60 DTE
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(f"DTE-based Iron Condor expiration: {dte_iron_condor.legs[0].expiration}")
    assert (
        pd.to_datetime(dte_iron_condor.legs[0].expiration)
        - pd.to_datetime("2024-09-06")
    ).days >= 60, "Expiration should be at least 60 days from entry"
    print("Test 45 passed: DTE-based Iron Condor created successfully")

    # Test 46: Straddle with DTE-based expiration
    print("Test 46: Straddle with DTE-based expiration")
    dte_straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike="ATM",
        expiration=30,  # 30 DTE
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(f"DTE-based Straddle expiration: {dte_straddle.legs[0].expiration}")
    assert (
        pd.to_datetime(dte_straddle.legs[0].expiration) - pd.to_datetime("2024-09-06")
    ).days >= 30, "Expiration should be at least 30 days from entry"
    print("Test 46 passed: DTE-based Straddle created successfully")

    # Test 47: Expiration selection with no valid options
    print("Test 47: Expiration selection with no valid options")
    try:
        invalid_dte_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration=1000,  # Unrealistically high DTE
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=entry_df,
        )
        assert False, "Should raise ValueError for invalid DTE"
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
        print("Test 47 passed: Appropriate error raised for invalid DTE")

    # Test 48: Comparison of date-based and DTE-based expiration selection
    print("Test 48: Comparison of date-based and DTE-based expiration selection")
    date_based_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",
        expiration="2024-10-25",  # Specific date
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    dte = (
        pd.to_datetime("2024-10-25 16:00:00") - pd.to_datetime("2024-09-06 15:30:00")
    ).days
    dte_based_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike="+2",
        short_strike="+0.3",
        expiration=dte,  # Equivalent DTE
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    print(f"Date-based expiration: {date_based_spread.legs[0].expiration}")
    print(f"DTE-based expiration: {dte_based_spread.legs[0].expiration}")
    assert pd.to_datetime(dte_based_spread.legs[0].expiration) >= pd.to_datetime(
        date_based_spread.legs[0].expiration
    ), "DTE-based expiration should be equal to or later than date-based expiration"
    print(
        "Test 48 passed: DTE-based expiration is equal to or later than date-based expiration"
    )

    print("\nAll DTE-based expiration selection tests completed!")

    print("\nTesting contract number synchronization:")

    # Test 49: Contract number synchronization
    print("Test 49: Contract number synchronization")
    sync_test_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(
        f"Initial contracts: Strategy - {sync_test_spread.contracts}, Legs - {[leg.contracts for leg in sync_test_spread.legs]}"
    )
    assert all(
        leg.contracts == 1 for leg in sync_test_spread.legs
    ), "Initial leg contracts should be 1"

    sync_test_spread.contracts = 2
    print(
        f"After update: Strategy - {sync_test_spread.contracts}, Legs - {[leg.contracts for leg in sync_test_spread.legs]}"
    )
    assert all(
        leg.contracts == 2 for leg in sync_test_spread.legs
    ), "All leg contracts should be updated to 2"

    print("Test 49 passed: Contract number synchronization works correctly")

    print("\nTesting contract number synchronization with different leg ratios:")

    # Test 50: Contract number synchronization with different leg ratios
    print("Test 50: Contract number synchronization with different leg ratios")
    butterfly_spread = OptionStrategy.create_butterfly(
        symbol="SPY",
        option_type="CALL",
        lower_strike=540,
        middle_strike=550,
        upper_strike=560,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(
        f"Initial contracts: Strategy - {butterfly_spread.contracts}, Legs - {[leg.contracts for leg in butterfly_spread.legs]}"
    )
    assert butterfly_spread.legs[0].contracts == 1, "Lower leg should have 1 contract"
    assert butterfly_spread.legs[1].contracts == 2, "Middle leg should have 2 contracts"
    assert butterfly_spread.legs[2].contracts == 1, "Upper leg should have 1 contract"

    butterfly_spread.contracts = 2
    print(
        f"After update: Strategy - {butterfly_spread.contracts}, Legs - {[leg.contracts for leg in butterfly_spread.legs]}"
    )
    assert butterfly_spread.legs[0].contracts == 2, "Lower leg should have 2 contracts"
    assert butterfly_spread.legs[1].contracts == 4, "Middle leg should have 4 contracts"
    assert butterfly_spread.legs[2].contracts == 2, "Upper leg should have 2 contracts"

    print(
        "Test 50 passed: Contract number synchronization works correctly with different leg ratios"
    )

    print("\nTesting straddle creation with leg ratios:")

    # Test 51: Straddle creation with default leg ratio
    print("Test 51: Straddle creation with default leg ratio")
    default_straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(
        f"Default straddle contracts: Strategy - {default_straddle.contracts}, Legs - {[leg.contracts for leg in default_straddle.legs]}"
    )
    assert all(
        leg.contracts == 1 for leg in default_straddle.legs
    ), "Default straddle should have 1 contract per leg"

    # Test 52: Straddle creation with custom leg ratio
    print("Test 52: Straddle creation with custom leg ratio")
    custom_straddle = OptionStrategy.create_straddle(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
        leg_ratio=2,
    )

    print(
        f"Custom straddle contracts: Strategy - {custom_straddle.contracts}, Legs - {[leg.contracts for leg in custom_straddle.legs]}"
    )
    assert all(
        leg.contracts == 2 for leg in custom_straddle.legs
    ), "Custom straddle should have 2 contracts per leg"

    # Test 53: Updating straddle contracts
    print("Test 53: Updating straddle contracts")
    custom_straddle.contracts = 2
    print(
        f"Updated straddle contracts: Strategy - {custom_straddle.contracts}, Legs - {[leg.contracts for leg in custom_straddle.legs]}"
    )
    assert all(
        leg.contracts == 4 for leg in custom_straddle.legs
    ), "Updated straddle should have 4 contracts per leg"

    print(
        "Straddle tests passed: Creation with leg ratios and contract updates work correctly"
    )

    print("\nTesting butterfly spread creation:")

    # Test 54: Butterfly spread creation
    print("Test 54: Butterfly spread creation")
    butterfly_spread = OptionStrategy.create_butterfly(
        symbol="SPY",
        option_type="CALL",
        lower_strike=540,
        middle_strike=550,
        upper_strike=560,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(
        f"Butterfly spread legs: {[(leg.strike, leg.contracts) for leg in butterfly_spread.legs]}"
    )
    assert len(butterfly_spread.legs) == 3, "Butterfly spread should have 3 legs"
    assert butterfly_spread.legs[0].contracts == 1, "Lower leg should have 1 contract"
    assert butterfly_spread.legs[1].contracts == 2, "Middle leg should have 2 contracts"
    assert butterfly_spread.legs[2].contracts == 1, "Upper leg should have 1 contract"
    assert butterfly_spread.legs[0].position_side == "BUY", "Lower leg should be BUY"
    assert butterfly_spread.legs[1].position_side == "SELL", "Middle leg should be SELL"
    assert butterfly_spread.legs[2].position_side == "BUY", "Upper leg should be BUY"

    # Test 55: Updating butterfly spread contracts
    print("Test 55: Updating butterfly spread contracts")
    butterfly_spread.contracts = 2
    print(
        f"Updated butterfly spread legs: {[(leg.strike, leg.contracts) for leg in butterfly_spread.legs]}"
    )
    assert butterfly_spread.legs[0].contracts == 2, "Lower leg should have 2 contracts"
    assert butterfly_spread.legs[1].contracts == 4, "Middle leg should have 4 contracts"
    assert butterfly_spread.legs[2].contracts == 2, "Upper leg should have 2 contracts"

    print("Butterfly spread tests passed: Creation and contract updates work correctly")

    print("\nTesting naked call creation and updates:")

    # Test 56: Naked call creation
    print("Test 56: Naked call creation")
    naked_call = OptionStrategy.create_naked_call(
        symbol="SPY",
        strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(f"Naked call leg: {naked_call.legs[0]}")
    assert len(naked_call.legs) == 1, "Naked call should have 1 leg"
    assert naked_call.legs[0].option_type == "CALL", "Naked call leg should be a CALL"
    assert naked_call.legs[0].position_side == "BUY", "Naked call leg should be BUY"
    assert naked_call.legs[0].contracts == 1, "Naked call should have 1 contract"

    # Test 57: Updating naked call
    print("Test 57: Updating naked call")
    naked_call.update("2024-09-06 15:45:00", update_df)
    print(f"Updated naked call P&L: ${naked_call.total_pl():.2f}")
    assert isinstance(naked_call.total_pl(), float), "P&L should be a float value"

    # Test 58: Naked call required capital
    print("Test 58: Naked call required capital")
    naked_call_capital = naked_call.get_required_capital()
    print(f"Naked call required capital: ${naked_call_capital:.2f}")
    assert naked_call_capital > 0, "Required capital for naked call should be positive"

    print(
        "Naked call tests passed: Creation, updates, and capital calculation work correctly"
    )

    print("\nTesting naked put creation and updates:")

    # Test 59: Naked put creation
    print("Test 59: Naked put creation")
    naked_put = OptionStrategy.create_naked_put(
        symbol="SPY",
        strike=540,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )

    print(f"Naked put leg: {naked_put.legs[0]}")
    assert len(naked_put.legs) == 1, "Naked put should have 1 leg"
    assert naked_put.legs[0].option_type == "PUT", "Naked put leg should be a PUT"
    assert naked_put.legs[0].position_side == "BUY", "Naked put leg should be BUY"
    assert naked_put.legs[0].contracts == 1, "Naked put should have 1 contract"

    # Test 60: Updating naked put
    print("Test 60: Updating naked put")
    naked_put.update("2024-09-06 15:45:00", update_df)
    print(f"Updated naked put P&L: ${naked_put.total_pl():.2f}")
    assert isinstance(naked_put.total_pl(), float), "P&L should be a float value"

    # Test 61: Naked put required capital
    print("Test 61: Naked put required capital")
    naked_put_capital = naked_put.get_required_capital()
    print(f"Naked put required capital: ${naked_put_capital:.2f}")
    assert naked_put_capital > 0, "Required capital for naked put should be positive"

    print(
        "Naked put tests passed: Creation, updates, and capital calculation work correctly"
    )

    # Test 62: Comparing naked call and put required capital
    print("Test 62: Comparing naked call and put required capital")
    print(f"Naked call required capital: ${naked_call_capital:.2f}")
    print(f"Naked put required capital: ${naked_put_capital:.2f}")
    assert (
        abs(naked_call_capital - naked_put_capital) < 100
    ), "Required capital for naked call and put should be similar"
    print("Required capital comparison test passed")

    print("\nAll tests completed!")
