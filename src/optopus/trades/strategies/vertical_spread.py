import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional
from ..option_spread import OptionStrategy

class VerticalSpread(OptionStrategy):
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
            commission (float, optional): Commission per contract per leg.
            exit_scheme (ExitConditionChecker, optional): Exit condition checker that determines when to close the position.
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.

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
