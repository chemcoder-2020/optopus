import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional, Type
from ..option_spread import OptionStrategy


class NakedCall(OptionStrategy):
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
        exit_scheme: Union[ExitConditionChecker, Type[ExitConditionChecker], dict] = {
            'class': DefaultExitCondition,
            'params': {
                'profit_target': 40,
                'exit_time_before_expiration': Timedelta(minutes=15),
                'window_size': 5
            }
        },
        **kwargs,
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
            exit_scheme (Union[ExitConditionChecker, Type[ExitConditionChecker], dict], optional): 
                The exit condition scheme to use. Can be:
                - An instance of ExitConditionChecker
                - A ExitConditionChecker class (will be instantiated with default params)
                - A dict containing:
                    - 'class': The ExitConditionChecker class
                    - 'params': Dict of parameters to pass to the constructor
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
            **kwargs,
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
        strategy.entry_bid, strategy.entry_ask = (
            strategy.current_bid,
            strategy.current_ask,
        )

        return strategy
