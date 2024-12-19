import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional
from ..option_spread import OptionStrategy
from .iron_condor import IronCondor


class IronButterfly(OptionStrategy):
    @classmethod
    def create_iron_butterfly(
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
        Create an iron butterfly option strategy.

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
            OptionStrategy: A iron butterfly strategy object.
        """
        if strategy_side not in ["DEBIT", "CREDIT"]:
            raise ValueError("Invalid strategy side. Must be 'DEBIT' or 'CREDIT'.")

        if strategy_side == "DEBIT":
            strategy = IronCondor.create_iron_condor(
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
            strategy = IronCondor.create_iron_condor(
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
        strategy.strategy_type = "Iron Butterfly"
        return strategy
