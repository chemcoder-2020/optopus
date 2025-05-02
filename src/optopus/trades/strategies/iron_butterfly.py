import pandas as pd
from pandas import Timestamp, Timedelta
import plotly.graph_objects as go
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional, Type
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
        max_extra_dte: int | None = None,
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
            exit_scheme (Union[ExitConditionChecker, Type[ExitConditionChecker], dict], optional):
                The exit condition scheme to use. Can be:
                - An instance of ExitConditionChecker
                - A ExitConditionChecker class (will be instantiated with default params)
                - A dict containing:
                    - 'class': The ExitConditionChecker class
                    - 'params': Dict of parameters to pass to the constructor
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
                max_extra_dte=max_extra_dte,
                **kwargs,
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
                max_extra_dte=max_extra_dte,
                **kwargs,
            )
        strategy.strategy_type = "Iron Butterfly"
        return strategy

    def plot_risk_profile(self):
        """Plot the risk profile curve for the iron butterfly strategy using Plotly.

        Shows the profit/loss diagram with breakeven points and key levels marked.
        """
        entry_premium = self.entry_net_premium
        put_leg = self.legs[0]
        call_leg = self.legs[2]
        middle_strike = self.legs[1].strike
        current_underlying_price = self.underlying_last
        is_credit = self.strategy_side == "CREDIT"
        current_pl = self.filter_pl

        price_range, pnl = self.generate_payoff_curve()

        # Calculate breakeven prices
        breakeven_lower = middle_strike - entry_premium
        breakeven_upper = middle_strike + entry_premium

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

        # Add strike lines
        for strike in [put_leg.strike, middle_strike, call_leg.strike]:
            fig.add_shape(
                type="line",
                x0=strike,
                y0=min(pnl),
                x1=strike,
                y1=max(pnl),
                line=dict(color="grey", dash="dashdot"),
                name=f"{strike} Strike",
            )

        # Add breakeven lines
        for be_price, be_type in [
            (breakeven_lower, "Lower"),
            (breakeven_upper, "Upper"),
        ]:
            fig.add_shape(
                type="line",
                x0=be_price,
                y0=min(pnl),
                x1=be_price,
                y1=max(pnl),
                line=dict(color="green" if is_credit else "red", dash="dot"),
                name=f"{be_type} Breakeven",
            )

        # Add annotations
        fig.add_annotation(
            x=breakeven_lower,
            y=0,
            text=f"Lower BE: {breakeven_lower:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
        )
        fig.add_annotation(
            x=breakeven_upper,
            y=0,
            text=f"Upper BE: {breakeven_upper:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
        )
        fig.add_annotation(
            x=current_underlying_price,
            y=current_pl,
            text=f"Current Price: {current_underlying_price:.2f}",
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
