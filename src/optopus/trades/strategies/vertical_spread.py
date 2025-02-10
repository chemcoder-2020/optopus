import pandas as pd
from pandas import Timestamp, Timedelta
import numpy as np
import plotly.graph_objects as go
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional, Type
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
            exit_scheme (Union[ExitConditionChecker, Type[ExitConditionChecker], dict], optional):
                The exit condition scheme to use. Can be:
                - An instance of ExitConditionChecker
                - A ExitConditionChecker class (will be instantiated with default params)
                - A dict containing:
                    - 'class': The ExitConditionChecker class
                    - 'params': Dict of parameters to pass to the constructor
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
            **kwargs,
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
        strategy.underlying_last = long_leg.underlying_last

        if strategy.entry_net_premium > spread_width:
            raise ValueError(
                "Entry net premium cannot be greater than the spread width."
            )

        return strategy

    def plot_risk_profile(self):
        """Plot the risk profile curve for the vertical spread strategy using Plotly.

        Shows the profit/loss diagram with breakeven points and key levels marked.
        """
        # Get strategy parameters
        entry_premium = self.entry_net_premium
        long_leg = self.legs[0]
        short_leg = self.legs[1]
        spread_width = abs(long_leg.strike - short_leg.strike)
        current_underlying_price = self.underlying_last
        is_call = long_leg.option_type == "CALL"
        is_credit = self.strategy_side == "CREDIT"

        # Generate price range for underlying
        min_strike = min(long_leg.strike, short_leg.strike)
        max_strike = max(long_leg.strike, short_leg.strike)
        price_range = np.linspace(
            min_strike - spread_width, max_strike + spread_width, 100
        )

        # Calculate profit/loss for each price point
        pnl = []
        for price in price_range:
            if is_call:
                long_payoff = max(price - long_leg.strike, 0)
                short_payoff = max(price - short_leg.strike, 0)
            else:
                long_payoff = min((price - long_leg.strike), 0)
                short_payoff = min((price - short_leg.strike), 0)

            net_payoff = (
                (-long_payoff + short_payoff + entry_premium) * 100 * self.contracts
            )
            pnl.append(net_payoff)

        # Calculate breakeven price
        if is_call:
            breakeven = (
                short_leg.strike + entry_premium
                if is_credit
                else long_leg.strike + entry_premium
            )
        else:
            breakeven = (
                short_leg.strike - entry_premium
                if is_credit
                else long_leg.strike - entry_premium
            )

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

        # Add breakeven line
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
        fig.add_shape(
            type="line",
            x0=long_leg.strike,
            y0=min(pnl),
            x1=long_leg.strike,
            y1=max(pnl),
            line=dict(color="grey", dash="dashdot"),
            name="Long Strike",
        )
        fig.add_shape(
            type="line",
            x0=short_leg.strike,
            y0=min(pnl),
            x1=short_leg.strike,
            y1=max(pnl),
            line=dict(color="grey", dash="dashdot"),
            name="Short Strike",
        )

        # Add annotations
        fig.add_annotation(
            x=breakeven,
            y=0,
            text=f"Breakeven: {breakeven:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
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
