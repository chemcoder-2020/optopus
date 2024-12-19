import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional
from ..option_spread import OptionStrategy


class IronCondor(OptionStrategy):
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
