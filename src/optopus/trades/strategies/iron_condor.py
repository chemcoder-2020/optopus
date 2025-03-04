import pandas as pd
from pandas import Timestamp, Timedelta
import numpy as np
import plotly.graph_objects as go
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional, Type
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
        exit_scheme: Union[ExitConditionChecker, Type[ExitConditionChecker], dict] = {
            "class": DefaultExitCondition,
            "params": {
                "profit_target": 40,
                "exit_time_before_expiration": Timedelta(minutes=15),
                "window_size": 5,
            },
        },
        **kwargs,
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
            exit_scheme (Union[ExitConditionChecker, Type[ExitConditionChecker], dict], optional):
                The exit condition scheme to use. Can be:
                - An instance of ExitConditionChecker
                - A ExitConditionChecker class (will be instantiated with default params)
                - A dict containing:
                    - 'class': The ExitConditionChecker class
                    - 'params': Dict of parameters to pass to the constructor
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
            **kwargs,
        )

        expiration_date = converter.get_closest_expiration(
            expiration, 
            max_extra_days=max_extra_days
        )

        put_short_strike_value = strategy.get_strike_value(
            converter, put_short_strike, expiration_date, "PUT",
            max_extra_days=max_extra_days
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
            max_extra_days=max_extra_days
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
        strategy.entry_bid, strategy.entry_ask = (
            strategy.current_bid,
            strategy.current_ask,
        )
        strategy.underlying_last = put_long_leg.underlying_last

        return strategy

    def plot_risk_profile(self):
        """Plot the risk profile curve for the iron condor strategy using Plotly.

        Shows the profit/loss diagram with breakeven points, strikes, and current price.
        """
        # Get strategy parameters
        entry_premium = self.entry_net_premium
        put_long = self.legs[0]
        put_short = self.legs[1]
        call_short = self.legs[2]
        call_long = self.legs[3]
        is_credit = self.strategy_side == "CREDIT"
        current_underlying_price = self.underlying_last
        current_pl = self.filter_pl

        price_range, pnl = self.generate_payoff_curve()

        # Calculate breakeven prices
        if is_credit:
            put_breakeven = put_short.strike - entry_premium
            call_breakeven = call_short.strike + entry_premium
        else:
            put_breakeven = put_long.strike + entry_premium
            call_breakeven = call_long.strike - entry_premium

        # Create plot
        fig = go.Figure()

        # Add P/L curve
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=pnl,
                mode="lines",
                name="P/L Curve",
                line=dict(color="royalblue", width=3),
                hovertemplate="Price: %{x}<br>P/L: %{y}",
            )
        )

        # Add breakeven lines
        for breakeven in [put_breakeven, call_breakeven]:
            fig.add_shape(
                type="line",
                x0=breakeven,
                y0=min(pnl),
                x1=breakeven,
                y1=max(pnl),
                line=dict(color="red", dash="dot"),
                name="Breakeven",
            )

        # Add strike lines
        strikes = [leg.strike for leg in self.legs]
        for strike in strikes:
            fig.add_shape(
                type="line",
                x0=strike,
                y0=min(pnl),
                x1=strike,
                y1=max(pnl),
                line=dict(color="grey", dash="dashdot"),
                name="Strike",
            )

        # Add current price annotation
        fig.add_annotation(
            x=current_underlying_price,
            y=current_pl,
            text=f"Current Price: {current_underlying_price:.2f}<br>P/L: ${current_pl:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
        )

        # Add annotations
        fig.add_annotation(
            x=put_breakeven,
            y=0,
            text=f"Put Breakeven: {put_breakeven:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-40,
        )

        fig.add_annotation(
            x=call_breakeven,
            y=0,
            text=f"Call Breakeven: {call_breakeven:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=-40,
        )

        fig.update_layout(
            title=f"{self.strategy_type} Risk Profile ({self.strategy_side})",
            xaxis_title="Underlying Price",
            yaxis_title="Profit/Loss ($)",
            hovermode="x unified",
            showlegend=True,
            margin=dict(l=50, r=50, b=50, t=50),
            height=600,
        )

        return fig.show()
