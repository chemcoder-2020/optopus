import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Union, Type
from .option_spread import OptionStrategy
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field
import numpy as np
from .entry_conditions import EntryConditionChecker, DefaultEntryCondition
from .external_entry_conditions import ExternalEntryConditionChecker
from ..metrics import (
    SharpeRatio,
    MaxDrawdown,
    RiskOfRuin,
    WinRate,
    ProfitFactor,
    CAGR,
    MonthlyReturn,
    YearlyReturn,
    PositiveMonthlyProbability,
    Volatility,
)


@dataclass
class Config:
    initial_capital: float
    max_positions: int
    max_positions_per_day: Optional[int] = None
    max_positions_per_week: Optional[int] = None
    position_size: float = 0.05
    ror_threshold: Optional[float] = None
    gain_reinvesting: bool = False
    verbose: bool = False
    ticker: str = None
    broker: str = None
    client_id: str = None
    client_secret: str = None
    redirect_uri: str = None
    token_file: str = None
    entry_condition: Union[EntryConditionChecker, Type[EntryConditionChecker], dict] = (
        field(default_factory=lambda: {"class": DefaultEntryCondition, "params": {}})
    )
    external_entry_condition: Union[
        ExternalEntryConditionChecker, Type[ExternalEntryConditionChecker], dict, None
    ] = field(default_factory=lambda: {"class": None, "params": {}})
    trade_type: str = None

    def __post_init__(self):
        """Initialize the config after dataclass creation."""
        self._initialize_entry_condition()
        self._initialize_external_entry_condition()

    def get(self, key, default=None):
        """Get an attribute with a default value if it does not exist."""
        return getattr(self, key, default)

    def _initialize_entry_condition(self):
        """Handle entry_condition initialization."""
        if isinstance(self.entry_condition, dict):
            # Case 1: Dictionary with class and parameters
            entry_class = self.entry_condition.get("class", DefaultEntryCondition)
            entry_params = self.entry_condition.get("params", {})
            self.entry_condition = entry_class(**entry_params)
        elif isinstance(self.entry_condition, type) and issubclass(
            self.entry_condition, EntryConditionChecker
        ):
            # Case 2: Just the class provided
            self.entry_condition = self.entry_condition()
        elif not isinstance(self.entry_condition, EntryConditionChecker):
            raise ValueError(
                "Invalid entry_condition format. Must be an instance, class, or dict with class and params"
            )

    def _initialize_external_entry_condition(self):
        """Handle external_entry_condition initialization."""
        if isinstance(self.external_entry_condition, dict):
            # Case 1: Dictionary with class and parameters
            ext_class = self.external_entry_condition.get("class")
            if ext_class is None:
                self.external_entry_condition = None
                return

            ext_params = self.external_entry_condition.get("params", {})
            self.external_entry_condition = ext_class(**ext_params)
        elif isinstance(self.external_entry_condition, type) and issubclass(
            self.external_entry_condition, ExternalEntryConditionChecker
        ):
            # Case 2: Just the class provided
            self.external_entry_condition = self.external_entry_condition()
        elif not isinstance(
            self.external_entry_condition, (ExternalEntryConditionChecker, type(None))
        ):
            raise ValueError(
                "Invalid external_entry_condition format. Must be an instance, class, dict with class and params, or None"
            )


class OptionBacktester:
    """Backtests option trading strategies."""

    def __init__(self, config: Config):
        """
        Initialize the backtester with a configuration.

        Args:
            config (Config): Configuration parameters for the backtester.

        Raises:
            ValueError: If the entry_condition is not an instance, class, or dict with class and params.
        """
        self.config = config
        self.capital = config.initial_capital
        self.allocation = config.initial_capital
        self.available_to_trade = config.initial_capital
        self.active_trades: List[OptionStrategy] = []
        self.closed_trades: List[OptionStrategy] = []
        self.last_update_time: Optional[datetime, pd.Timestamp] = (
            None  # pd.Timestamp.now(tz="America/New_York")
        )
        self.trades_entered_today = 0
        self.trades_entered_this_week = 0
        self.performance_data = []
        self.context = {"indicators": {}}

    def update(self, current_time: datetime, option_chain_df: pd.DataFrame) -> None:
        """
        Update the backtester with the current time and option chain data.

        Args:
            current_time (datetime): Current time for the update.
            option_chain_df (pd.DataFrame): DataFrame containing the option chain data.
        """
        if not hasattr(self, "context"):
            self.context = {"indicators": {}}
        if "indicators" not in self.context:
            self.context["indicators"] = {}

        self.context.update(
            {
                "option_chain_df": option_chain_df,
            }
        )

        try:
            current_time = pd.to_datetime(current_time)
            self.last_update_time = current_time

            # Update all active trades
            if self.active_trades != []:
                for trade in self.active_trades:
                    trade_update_success = trade.update(current_time, option_chain_df)
                    if not trade_update_success:
                        logger.warning(
                            f"Trade {trade} update failed at {current_time}, due to spike in option chain."
                        )
                    elif trade.status == "CLOSED":
                        self.close_trade(trade)

            self._update_trade_counts()

        except Exception as e:
            logger.error(f"Error updating backtester: {str(e)}")

    def add_spread(self, new_spread: OptionStrategy) -> bool:
        """
        Add a new option spread to the backtester.

        Args:
            new_spread (OptionStrategy): The new option spread to add.

        Returns:
            bool: True if the spread was added, False otherwise.
        """
        try:
            if not self._can_add_spread(new_spread):
                logger.info(
                    f"Rejecting {new_spread.symbol} {new_spread.strategy_type} - entry conditions not met"
                )
                return False

            required_capital = new_spread.get_required_capital()
            # Calculate and store entry delta
            new_spread.entry_delta = new_spread.current_delta()
            new_spread.manager = self
            self.active_trades.append(new_spread)
            self.available_to_trade -= required_capital
            self._update_trade_counts()
            logger.success(
                f"Added {new_spread.symbol} {new_spread.strategy_type} (Contracts: {new_spread.contracts}, Required Capital: ${required_capital:.2f}, Available: ${self.available_to_trade:.2f})"
            )
            return True
        except Exception as e:
            logger.error(f"Error adding spread: {str(e)}")
            return False

    def _can_add_spread(self, new_spread: OptionStrategy) -> bool:
        """
        Check if a new spread can be added based on entry conditions.

        Args:
            new_spread (OptionStrategy): The new option spread to check.

        Returns:
            bool: True if the spread can be added, False otherwise.
        """
        if self.capital <= 0:
            logger.warning(
                f"Cannot add {new_spread.symbol} {new_spread.strategy_type}: no capital left"
            )
            return False

        # Check external entry conditions first if configured

        if self.config.external_entry_condition is not None:
            try:
                external_met = self.config.external_entry_condition.should_enter(
                    time=self.last_update_time, strategy=new_spread, manager=self
                )
            except Exception as e:
                logger.error(
                    f"Error checking external entry conditions for {new_spread.symbol} {new_spread.strategy_type}: {str(e)}"
                )
                external_met = False
            if not external_met:
                logger.info(
                    f"External conditions not met for {new_spread.symbol} {new_spread.strategy_type}"
                )
                # return False
            logger.info(
                f"External conditions met for {new_spread.symbol} {new_spread.strategy_type}"
            )
        else:
            external_met = True
            logger.info(
                f"No external entry conditions configured for {new_spread.symbol} {new_spread.strategy_type}"
            )

        # Check standard entry conditions (required for both cases)
        try:
            standard_met = self.config.entry_condition.should_enter(
                strategy=new_spread, manager=self, time=self.last_update_time
            )
        except Exception as e:
            logger.error(
                f"Error checking standard entry conditions for {new_spread.symbol} {new_spread.strategy_type}: {str(e)}"
            )
            standard_met = False

        if not standard_met:
            logger.info(
                f"Standard conditions not met for {new_spread.symbol} {new_spread.strategy_type}"
            )
            # return False

        if external_met and standard_met:
            logger.info(
                f"All conditions met for {new_spread.symbol} {new_spread.strategy_type}"
            )
            return True
        else:
            logger.info(
                f"Conditions not met for {new_spread.symbol} {new_spread.strategy_type}"
            )
            return False

        # return True

    def close_trade(self, trade: OptionStrategy) -> None:
        """
        Close a trade and update capital and allocation.

        Args:
            trade (OptionStrategy): The trade to close.
        """
        self.active_trades.remove(trade)
        self.closed_trades.append(trade)

        pl_change = trade.filter_pl if not np.isnan(trade.filter_pl) else 0
        recovered_capital = trade.get_required_capital()

        self.capital += pl_change
        self.available_to_trade += recovered_capital

        assert not np.isnan(
            self.capital
        ), f"Capital is NaN: {self.capital} at {trade.exit_time}"

        # Update allocation if gain_reinvesting is True
        if self.config.gain_reinvesting:
            new_allocation = max(self.capital, self.allocation)
            added_allocation = new_allocation - self.allocation
            self.allocation = new_allocation
            self.available_to_trade += added_allocation

        self._update_trade_counts()

    def _update_trade_counts(self) -> None:
        """
        Update the count of trades entered today and this week.
        """
        # Count trades entered on the same calendar date (including closed ones)
        if self.last_update_time:
            self.trades_entered_today = sum(
                1
                for trade in self.active_trades + self.closed_trades
                if trade.entry_time.date() == self.last_update_time.date()
            )

        # Weekly count remains active trades only
        self.trades_entered_this_week = sum(
            1
            for trade in self.active_trades
            if trade.entry_time.isocalendar()[1]
            != self.last_update_time.isocalendar()[1]
        )

    def _check_conflict(self, new_spread: OptionStrategy) -> bool:
        """
        Check if a new spread conflicts with any active trades.

        Args:
            new_spread (OptionStrategy): The new option spread to check.

        Returns:
            bool: True if there is a conflict, False otherwise.
        """
        return any(
            existing_spread.conflicts_with(new_spread)
            for existing_spread in self.active_trades
        )

    def _check_ror(self, spread: OptionStrategy) -> bool:
        """
        Check if a spread meets the return over risk threshold.

        Args:
            spread (OptionStrategy): The option spread to check.

        Returns:
            bool: True if the spread meets the threshold, False otherwise.
        """
        return spread.return_over_risk() >= self.config.ror_threshold

    def get_total_pl(self) -> float:
        """
        Calculate the total profit and loss from all trades.

        Returns:
            float: Total profit and loss.
        """
        return np.nansum(
            [trade.filter_pl for trade in self.active_trades + self.closed_trades]
        )

    def get_closed_pl(self) -> float:
        """
        Calculate the profit and loss from closed trades.

        Returns:
            float: Profit and loss from closed trades.
        """
        return np.nansum([trade.filter_pl for trade in self.closed_trades])

    def get_open_positions(self) -> int:
        """
        Get the number of open positions.

        Returns:
            int: Number of open positions.
        """
        return len(self.active_trades)

    def get_closed_positions(self) -> int:
        """
        Get the number of closed positions.

        Returns:
            int: Number of closed positions.
        """
        return len(self.closed_trades)

    def _record_performance_data(
        self, current_time: datetime, option_chain_df: pd.DataFrame
    ) -> None:
        """
        Record performance data at the current time.

        Args:
            current_time (datetime): Current time for the performance data.
            option_chain_df (pd.DataFrame): DataFrame containing the option chain data.
        """
        total_pl = self.get_total_pl()
        closed_pl = self.get_closed_pl()
        active_positions = len(self.active_trades)
        underlying_last = (
            option_chain_df["UNDERLYING_LAST"].iloc[0]
            if "UNDERLYING_LAST" in option_chain_df.columns
            else 0
        )

        self.performance_data.append(
            {
                "time": current_time,
                "total_pl": total_pl,
                "closed_pl": closed_pl,
                "underlying_last": underlying_last,
                "active_positions": active_positions,
                "indicators": {**self.context.get("indicators", {})},
            }
        )

    def plot_performance(self, interactive: bool = False):
        """
        Generate and display performance visualizations.

        Args:
            interactive (bool): If True, shows interactive Plotly version.
                Default False shows matplotlib version.
        """
        if not self.performance_data:
            logger.warning("No performance data available for plotting.")
            return

        df = pd.DataFrame(self.performance_data)

        if interactive:
            return self._plot_interactive_performance(df)
        else:
            return self._plot_static_performance(df)

    def _plot_static_performance(self, df):
        """Original matplotlib implementation"""
        if df["indicators"].dropna().empty:
            logger.warning("No indicators data available for plotting.")
            indicators = None
            number_of_indicators = 0
        else:
            indicators = df["indicators"].apply(lambda x: pd.Series(x))
            indicators["time"] = df["time"]
            indicators.set_index("time", inplace=True)
            number_of_indicators = len(indicators.columns)

        df.set_index("time", inplace=True)

        # Calculate drawdown
        df["peak"] = df["total_pl"].cummax()
        df["drawdown"] = df["peak"] - df["total_pl"]

        # Create subplots
        fig, axes = plt.subplots(
            5 + number_of_indicators, 1, figsize=(12, 24), sharex=True
        )

        # Plot Total P/L
        axes[0].plot(df.index, df["total_pl"], label="Total P/L")
        axes[0].set_title("Total P/L")
        axes[0].set_ylabel("P/L ($)")
        axes[0].legend()

        # Plot Closed P/L
        axes[1].plot(df.index, df["closed_pl"], label="Closed P/L")
        axes[1].set_title("Closed P/L")
        axes[1].set_ylabel("P/L ($)")
        axes[1].legend()

        # Plot Drawdown
        axes[2].fill_between(df.index, df["drawdown"], label="Drawdown")
        axes[2].set_title("Drawdown")
        axes[2].set_ylabel("Drawdown ($)")
        axes[2].legend()

        # Plot Underlying Price
        axes[3].plot(df.index, df["underlying_last"], label="Underlying Price")
        axes[3].set_title("Underlying Price")
        axes[3].set_ylabel("Price ($)")
        axes[3].set_xlabel("Date")
        axes[3].legend()

        # Plot Active Positions
        axes[4].plot(df.index, df["active_positions"], label="Active Positions")
        axes[4].set_title("Active Positions")
        axes[4].set_ylabel("Positions")
        axes[4].set_xlabel("Date")
        axes[4].legend()

        # Plot Indicators
        if indicators is not None:
            for i, col in enumerate(indicators.columns):
                axes[5 + i].plot(indicators.index, indicators[col], label=col)
                axes[5 + i].set_title(f"Indicator: {col}")
                axes[5 + i].set_ylabel(f"{col}")
                axes[5 + i].set_xlabel("Date")
                axes[5 + i].legend()

        plt.tight_layout()
        plt.show()
        return fig

    def _plot_interactive_performance(self, df):
        """Create interactive Plotly version with shared x-axis and crosshair"""
        # Prepare data
        df = df.copy()
        df.set_index("time", inplace=True)
        df["peak"] = df["total_pl"].cummax()
        df["drawdown"] = df["peak"] - df["total_pl"]

        # Check if we have indicators
        has_indicators = not df["indicators"].dropna().empty
        num_indicators = 0
        if has_indicators:
            for val in df["indicators"]:
                if isinstance(val, (dict, list)):
                    num_indicators = len(val)
                    break
        
        # Create subplots dynamically based on indicators
        subplot_titles = [
            "Total P/L", 
            "Closed P/L",
            "Drawdown",
            "Underlying Price",
            "Active Positions"
        ]
        
        if has_indicators:
            indicators = df["indicators"].apply(pd.Series)
            subplot_titles += [f"Indicator: {col}" for col in indicators.columns]
        
        fig = make_subplots(
            rows=len(subplot_titles),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        # Add main traces
        fig.add_trace(
            go.Scatter(x=df.index, y=df["total_pl"], name="Total P/L"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df["closed_pl"], name="Closed P/L"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df["drawdown"], fill='tozeroy', name="Drawdown"),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df["underlying_last"], name="Underlying Price"),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df["active_positions"], name="Active Positions"),
            row=5, col=1
        )

        # Add indicators if available
        if has_indicators:
            for i, col in enumerate(indicators.columns):
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=indicators[col], 
                        name=f"Indicator: {col}",
                        showlegend=False
                    ),
                    row=6+i,  # Start at row 6
                    col=1
                )
                fig.update_yaxes(title_text=col, row=6+i, col=1)

        # Formatting and crosshair configuration
        fig.update_layout(
            height=900 + (200 * num_indicators),  # Dynamic height based on indicators
            width=900,
            title_text="Trading Performance",
            hovermode="x unified",
            spikedistance=1000,
            hoverdistance=100
        )
        
        # Apply crosshair to all subplots
        for i in range(1, len(subplot_titles)+1):
            fig.update_xaxes(
                showspikes=True,
                spikemode="across",
                spikedash="solid",
                row=i, 
                col=1
            )
        fig.update_traces(xaxis="x1")
        fig.show()
        return fig

    def get_closed_trades_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame of closed trades with various attributes.

        Returns:
            pd.DataFrame: DataFrame containing closed trades information.
        """
        closed_trades_data = []
        cumulative_pl = 0

        for trade in self.closed_trades:
            pl = trade.filter_pl
            cumulative_pl += pl

            trade_data = {
                "symbol": trade.symbol,
                "strategy_type": trade.strategy_type,
                "entry_time": trade.entry_time,
                "entry_dte": trade.entry_dte,
                "exit_time": trade.exit_time,
                "exit_dte": trade.exit_dte,
                "contracts": trade.contracts,
                "entry_underlying_last": trade.entry_underlying_last,
                "exit_underlying_last": trade.exit_underlying_last,
                "entry_net_premium": trade.entry_net_premium,
                "entry_bid": trade.entry_bid,
                "entry_ask": trade.entry_ask,
                "exit_net_premium": trade.exit_net_premium,
                "entry_delta": trade.entry_delta,
                "closed_pl": pl,
                "cumulative_pl": cumulative_pl,
                "return_percentage": trade.return_percentage(),
                "return_over_risk": pl / trade.get_required_capital(),
                "dit": trade.DIT,
                "won": trade.won,
            }

            # Add leg-specific information
            for i, leg in enumerate(trade.legs):
                trade_data.update(
                    {
                        f"leg{i+1}_type": leg.option_type,
                        f"leg{i+1}_strike": leg.strike,
                        f"leg{i+1}_position": leg.position_side,
                        f"leg{i+1}_entry_price": leg.entry_price,
                        f"leg{i+1}_entry_bid": leg.entry_bid,
                        f"leg{i+1}_entry_ask": leg.entry_ask,
                        f"leg{i+1}_exit_price": leg.current_price,
                    }
                )

            closed_trades_data.append(trade_data)

        return pd.DataFrame(closed_trades_data)

    def calculate_kelly_criterion(
        self, n: Optional[int] = None, fractional_factor: Optional[float] = None
    ) -> float:
        """
        Calculate the Kelly Criterion percentage for position sizing based on recent trades.

        Args:
            n: Number of recent trades to consider (None for all trades)
            fractional_factor: Fraction of Kelly to recommend (e.g. 0.5 for half Kelly)

        Returns:
            float: Recommended position size percentage (0-1)
        """
        try:
            trades = self.closed_trades[-n:] if n else self.closed_trades
            if not trades:
                return 0.0

            wins = [t for t in trades if t.won]
            losses = [t for t in trades if not t.won]

            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t.total_pl() for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t.total_pl()) for t in losses]) if losses else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

            kelly = (
                (win_rate - (1 - win_rate) / win_loss_ratio)
                if win_loss_ratio != 0
                else 0
            )
            kelly = max(min(kelly, 1.0), 0.0)  # clamp between 0-100%

            if fractional_factor:
                kelly *= fractional_factor

            return kelly

        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {str(e)}")
            return 0.0

    def calculate_performance_metrics(self) -> Optional[dict]:
        """
        Calculate various performance metrics.

        Returns:
            dict: Dictionary containing performance metrics.
        """
        if not self.performance_data:
            logger.warning("No performance data available for metric calculation.")
            return None

        df = pd.DataFrame(self.performance_data)
        df.set_index("time", inplace=True)

        closed_trades_df = self.get_closed_trades_df()
        try:
            trade_returns_per_allocation = (
                closed_trades_df["closed_pl"] / self.allocation
            )
        except KeyError:
            trade_returns_per_allocation = np.nan

        try:
            average_dit = closed_trades_df["dit"].mean()
        except KeyError:
            average_dit = np.nan

        try:
            average_dit_spread = closed_trades_df["dit"].std()
        except KeyError:
            average_dit_spread = np.nan

        try:
            average_exit_dte = closed_trades_df["exit_dte"].mean()
        except KeyError:
            average_exit_dte = np.nan

        drawdown_calculator = MaxDrawdown()
        max_drawdown_result = drawdown_calculator.calculate(
            df["total_pl"].values, self.allocation
        )
        max_drawdown_dollars = max_drawdown_result["max_drawdown_dollars"]
        max_drawdown_percentage = max_drawdown_result["max_drawdown_percentage"]
        max_drawdown_percentage_from_peak = max_drawdown_result[
            "max_drawdown_percentage_from_peak"
        ]

        metrics = {
            "sharpe_ratio": None,
            "profit_factor": None,
            "cagr": None,
            "avg_monthly_pl": None,
            "avg_monthly_pl_nonzero": None,
            "win_rate": None,
            "risk_of_ruin": None,
            "return_over_risk": None,
            "probability_of_positive_monthly_pl": None,
            "probability_of_positive_monthly_closed_pl": None,
            "max_drawdown_dollars": max_drawdown_dollars,
            "max_drawdown_percentage": max_drawdown_percentage,
            "max_drawdown_percentage_from_peak": max_drawdown_percentage_from_peak,
            "average_dit": average_dit,
            "average_dit_spread": average_dit_spread,
            "average_exit_dte": average_exit_dte,
        }

        try:
            # Calculate daily P/L changes from performance data
            daily_pl = df["total_pl"].resample("B").last().ffill()
            daily_returns = daily_pl.diff().dropna() / self.config.initial_capital
            sharpe_calculator = SharpeRatio()
            metrics["sharpe_ratio"] = sharpe_calculator.calculate(
                daily_returns.values, risk_free_rate=0.02
            )["sharpe_ratio"]
            
            # Calculate volatility using same daily returns
            volatility_calculator = Volatility()
            metrics["annualized_volatility"] = volatility_calculator.calculate(
                daily_returns.values
            )["annualized_volatility"]
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio/Volatility: {str(e)}")
        try:
            pf_calculator = ProfitFactor()
            metrics["profit_factor"] = pf_calculator.calculate(
                trade_returns_per_allocation.values
            )["profit_factor"]
        except Exception as e:
            logger.error(f"Error calculating Profit Factor: {str(e)}")
        try:
            # Calculate CAGR
            start_value = self.config.initial_capital
            end_value = start_value + df["closed_pl"].iloc[-1]
            start_time = df.index[0]
            end_time = df.index[-1]

            cagr_calculator = CAGR()
            cagr_result = cagr_calculator.calculate(
                start_value, end_value, start_time, end_time
            )
            metrics["cagr"] = cagr_result["cagr"]
        except Exception as e:
            logger.error(f"Error calculating CAGR: {str(e)}")
        try:
            monthly_return_calculator = MonthlyReturn()
            metrics["avg_monthly_pl"] = monthly_return_calculator.calculate(
                closed_trades_df.set_index("exit_time")["closed_pl"]
            )["avg_monthly_pl"]
        except Exception as e:
            logger.error(f"Error calculating Average Monthly P/L: {str(e)}")
        try:
            metrics["avg_monthly_pl_nonzero"] = monthly_return_calculator.calculate(
                closed_trades_df.set_index("exit_time")["closed_pl"], non_zero_only=True
            )["avg_monthly_pl"]
        except Exception as e:
            logger.error(
                f"Error calculating Average Monthly P/L (Non-Zero Months): {str(e)}"
            )
        try:
            yearly_return_calculator = YearlyReturn()
            metrics["median_yearly_pl"] = yearly_return_calculator.calculate(
                closed_trades_df.set_index("exit_time")["closed_pl"], non_zero_only=True
            )["median_yearly_pl"]
            metrics["median_yearly_return"] = (
                metrics["median_yearly_pl"] / self.config.initial_capital
            )
        except Exception as e:
            logger.error(f"Error calculating Average Monthly P/L: {str(e)}")
        try:
            wins = [t.won for t in self.closed_trades]
            winrate_calculator = WinRate()
            metrics["win_rate"] = winrate_calculator.calculate(np.array(wins))[
                "win_rate"
            ]
        except Exception as e:
            logger.error(f"Error calculating Win Rate: {str(e)}")
        try:
            # Calculate daily P/L changes from performance data
            daily_pl = df["total_pl"].resample("B").last().ffill().dropna()

            risk_of_ruin_calculator = RiskOfRuin()
            risk_result = risk_of_ruin_calculator.calculate(
                daily_pl=daily_pl.values,
                initial_balance=self.config.initial_capital,
                distribution="histogram",
            )
            metrics["risk_of_ruin"] = risk_result["risk_of_ruin"]
        except Exception as e:
            logger.error(f"Error calculating Risk of Ruin: {str(e)}")
        try:
            prob_calculator = PositiveMonthlyProbability()
            metrics["probability_of_positive_monthly_pl"] = prob_calculator.calculate(
                df["total_pl"]
            )["positive_monthly_probability"]
            metrics["probability_of_positive_monthly_closed_pl"] = (
                prob_calculator.calculate(df["closed_pl"])[
                    "positive_monthly_probability"
                ]
            )
        except Exception as e:
            logger.error(
                f"Error calculating Probability of Positive Monthly Closed P/L: {str(e)}"
            )

        try:
            metrics["return_over_risk"] = closed_trades_df["return_over_risk"].mean()
        except KeyError:
            logger.error("Error calculating Return Over Risk.")

        return metrics

    def update_config(self, **kwargs) -> bool:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update.
                    Only existing attributes in Config will be updated.

        Returns:
            bool: True if any parameters were updated, False otherwise.
        """
        updated = False
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # Update attribute
                if key in ["initial_capital", "allocation", "position_size"]:
                    value = float(value)
                setattr(self.config, key, value)
                # Update related attributes that depend on config
                if key == "initial_capital" or key == "allocation":
                    new_allocation = value
                    added_allocation = new_allocation - self.allocation
                    self.allocation = value
                    self.available_to_trade += added_allocation
                    self.capital += added_allocation
                elif key == "position_size":
                    # Validate position size is between 0 and 1
                    if not 0 < value <= 1:
                        logger.warning(
                            f"Invalid position_size {value}. Must be between 0 and 1."
                        )
                        continue
                updated = True
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        return updated

    def print_performance_summary(self):
        """
        Print a summary of performance metrics.
        """
        metrics = self.calculate_performance_metrics()
        if metrics:
            print("\nPerformance Summary:")
            if metrics.get("sharpe_ratio") is not None:
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            if metrics.get("annualized_volatility") is not None:
                print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
            if metrics.get("profit_factor") is not None:
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            if metrics.get("cagr") is not None:
                print(f"CAGR: {metrics['cagr']:.2%}")
            if metrics.get("median_yearly_pl") is not None:
                print(f"Median Yearly P/L: ${metrics['median_yearly_pl']:.2f}")
            if metrics.get("median_yearly_return") is not None:
                print(f"Median Yearly Return: {metrics['median_yearly_return']:.2%}")
            if metrics.get("avg_monthly_pl") is not None:
                print(f"Average Monthly P/L: ${metrics['avg_monthly_pl']:.2f}")
            if metrics.get("avg_monthly_pl_nonzero") is not None:
                print(
                    f"Average Monthly P/L (Non-Zero Months): ${metrics['avg_monthly_pl_nonzero']:.2f}"
                )
            if metrics.get("return_over_risk") is not None:
                print(f"Average Return over Risk: {metrics['return_over_risk']:.2%}")
            if metrics.get("probability_of_positive_monthly_pl") is not None:
                print(
                    f"Probability of Positive Monthly P/L: {metrics['probability_of_positive_monthly_pl']:.2%}"
                )
            if metrics.get("probability_of_positive_monthly_closed_pl") is not None:
                print(
                    f"Probability of Positive Monthly Closed P/L: {metrics['probability_of_positive_monthly_closed_pl']:.2%}"
                )
            if metrics.get("win_rate") is not None:
                print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(
                f"Kelly Criterion Recommendation: {self.calculate_kelly_criterion():.2%}"
            )
            if metrics.get("risk_of_ruin") is not None:
                print(f"Risk of Ruin: {metrics['risk_of_ruin']:.2%}")
            if metrics.get("max_drawdown_dollars") is not None:
                print(f"Max Drawdown (Dollars): ${metrics['max_drawdown_dollars']:.2f}")
            if metrics.get("max_drawdown_percentage") is not None:
                print(
                    f"Max Drawdown (Percentage): {metrics['max_drawdown_percentage']:.2%}"
                )
            if metrics.get("max_drawdown_percentage_from_peak") is not None:
                print(
                    f"Max Drawdown From Peak (Percentage): {metrics['max_drawdown_percentage_from_peak']:.2%}"
                )
            if metrics.get("average_dit") is not None:
                print(f"Average Days in Trade: {metrics['average_dit']:.2f}")

            # Print time range of closed positions
            if self.closed_trades:
                first_entry_time = min(trade.entry_time for trade in self.closed_trades)
                last_exit_time = max(trade.exit_time for trade in self.closed_trades)
                print(
                    f"Time Range of Closed Positions: {first_entry_time} to {last_exit_time}"
                )
        else:
            print("No performance data available for summary.")
