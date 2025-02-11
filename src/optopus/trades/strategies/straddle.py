import pandas as pd
from pandas import Timestamp, Timedelta
from ..option_leg import OptionLeg
from ..exit_conditions import DefaultExitCondition, ExitConditionChecker
from ..option_chain_converter import OptionChainConverter
from typing import Union, Tuple, Optional, Type
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
            exit_scheme (Union[ExitConditionChecker, Type[ExitConditionChecker], dict], optional): 
                The exit condition scheme to use. Can be:
                - An instance of ExitConditionChecker
                - A ExitConditionChecker class (will be instantiated with default params)
                - A dict containing:
                    - 'class': The ExitConditionChecker class
                    - 'params': Dict of parameters to pass to the constructor
                Defaults to DefaultExitCondition with 40% profit target, 15-minute buffer before expiration, and 5-minute window size.
            strategy_side (str, optional): The strategy side ('DEBIT' or 'CREDIT'). Defaults to 'DEBIT'.

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
            **kwargs,
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
            "BUY" if strategy_side == "DEBIT" else "SELL",
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
            "BUY" if strategy_side == "DEBIT" else "SELL",
            commission=commission,
        )

        strategy.strategy_side = strategy_side

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
        strategy.underlying_last = put_leg.underlying_last

        return strategy

    def plot_risk_profile(self):
        """Plot the risk profile curve for the straddle strategy using Plotly.

        Shows the profit/loss diagram with breakeven points and key levels marked.
        """
        entry_premium = abs(self.entry_net_premium)
        call_leg = self.legs[0]
        put_leg = self.legs[1]
        current_underlying_price = self.underlying_last
        is_debit = self.strategy_side == "DEBIT"
        current_pl = self.filter_pl

        price_range, pnl = self.generate_payoff_curve()

        # Calculate breakeven prices
        breakeven_up = call_leg.strike + entry_premium
        breakeven_down = put_leg.strike - entry_premium

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

        # Add breakeven lines
        for be_price, direction in [(breakeven_up, "Upper"), (breakeven_down, "Lower")]:
            fig.add_shape(
                type="line",
                x0=be_price,
                y0=min(pnl),
                x1=be_price,
                y1=max(pnl),
                line=dict(color="green" if is_debit else "red", dash="dot"),
                name=f"{direction} Breakeven",
            )

        # Add annotations
        fig.add_annotation(
            x=breakeven_up,
            y=0,
            text=f"Upper BE: {breakeven_up:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
        )
        fig.add_annotation(
            x=breakeven_down,
            y=0,
            text=f"Lower BE: {breakeven_down:.2f}",
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
