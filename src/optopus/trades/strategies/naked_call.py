import pandas as pd
from pandas import Timestamp, Timedelta
import plotly.graph_objects as go
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

        expiration_date = converter.get_closest_expiration(
            expiration,
            max_extra_days=max_extra_days
        )

        # Get strike price using the converter
        strike_value = cls.get_strike_value(
            converter, 
            strike, 
            expiration_date, 
            "CALL",
            max_extra_days=max_extra_days
        )

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
        strategy.underlying_last = call_leg.underlying_last

        return strategy

    def plot_risk_profile(self):
        """Plot the risk profile curve for the naked call strategy using Plotly.

        Shows the profit/loss diagram with breakeven point and key levels marked.
        """
        entry_premium = self.entry_net_premium
        call_leg = self.legs[0]
        current_underlying_price = self.underlying_last
        is_credit = self.strategy_side == "CREDIT"
        current_pl = self.filter_pl

        price_range, pnl = self.generate_payoff_curve()

        # Calculate key levels
        breakeven_price = call_leg.strike + entry_premium
        max_profit = entry_premium if is_credit else None
        max_loss = "Unlimited" if self.strategy_side == "SHORT" else None

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

        # Add strike line
        fig.add_shape(
            type="line",
            x0=call_leg.strike,
            y0=min(pnl),
            x1=call_leg.strike,
            y1=max(pnl),
            line=dict(color="grey", dash="dashdot"),
            name="Strike Price",
        )

        # Add breakeven line
        fig.add_shape(
            type="line",
            x0=breakeven_price,
            y0=min(pnl),
            x1=breakeven_price,
            y1=max(pnl),
            line=dict(color="green" if is_credit else "red", dash="dot"),
            name="Breakeven",
        )

        # Add annotations
        fig.add_annotation(
            x=breakeven_price,
            y=0,
            text=f"Breakeven: {breakeven_price:.2f}",
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
