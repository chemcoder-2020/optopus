import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional
from ..option_spread import OptionStrategy


class Straddle(OptionStrategy):
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
        strategy.entry_bid, strategy.entry_ask = (
            strategy.current_bid,
            strategy.current_ask,
        )

        return strategy
