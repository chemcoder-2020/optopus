import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Type
from .option_spread import OptionStrategy
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import gaussian_kde
from .entry_conditions import EntryConditionChecker, DefaultEntryCondition
from .external_entry_conditions import ExternalEntryConditionChecker
from ..metrics import (
    SharpeRatio,
    MaxDrawdown,
    RiskOfRuin,
    TotalReturn,
    AnnualizedReturn,
    WinRate,
    ProfitFactor,
    CAGR,
    MonthlyReturn,
    PositiveMonthlyProbability,
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
        self.last_update_time: Optional[datetime, pd.Timestamp] = None
        self.trades_entered_today = 0
        self.trades_entered_this_week = 0
        self.performance_data = []

    def update(self, current_time: datetime, option_chain_df: pd.DataFrame) -> None:
        """
        Update the backtester with the current time and option chain data.

        Args:
            current_time (datetime): Current time for the update.
            option_chain_df (pd.DataFrame): DataFrame containing the option chain data.
        """

        try:
            current_time = pd.to_datetime(current_time)

            # Update all active trades
            for trade in self.active_trades:
                trade_update_success = trade.update(current_time, option_chain_df)
                if not trade_update_success:
                    logger.warning(
                        f"Trade {trade} update failed at {current_time}, due to spike in option chain."
                    )
                elif trade.status == "CLOSED":
                    self.close_trade(trade)

            self._update_trade_counts()
            self.last_update_time = current_time

            # Record performance data after update
            self._record_performance_data(current_time, option_chain_df)
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
                return False

            required_capital = new_spread.get_required_capital()
            # Calculate and store entry delta
            new_spread.entry_delta = new_spread.current_delta()
            self.active_trades.append(new_spread)
            self.available_to_trade -= required_capital
            self._update_trade_counts()
            logger.info(f"Added new spread: {new_spread}")
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
            logger.warning("Cannot add spread: no capital left.")
            return False

        # Check external entry conditions first if configured
        if self.config.external_entry_condition is not None:
            external_met = self.config.external_entry_condition.should_enter(
                time=self.last_update_time, strategy=new_spread, manager=self
            )
            if not external_met:
                logger.info("External entry conditions not met")
                return False
            logger.info("External entry conditions met")

        # Check standard entry conditions (required for both cases)
        standard_met = self.config.entry_condition.should_enter(
            new_spread, self, self.last_update_time
        )

        if not standard_met:
            logger.info("Standard entry conditions not met")
            return False

        logger.info("All entry conditions met")
        return True

    def close_trade(self, trade: OptionStrategy) -> None:
        """
        Close a trade and update capital and allocation.

        Args:
            trade (OptionStrategy): The trade to close.
        """
        self.active_trades.remove(trade)
        self.closed_trades.append(trade)

        pl_change = trade.total_pl()
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
        self.trades_entered_today = sum(
            1 for trade in self.active_trades if trade.DIT == 0
        )
        self.trades_entered_this_week = sum(
            1 for trade in self.active_trades if trade.DIT < 7
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
        return sum(
            trade.total_pl() for trade in self.active_trades + self.closed_trades
        )

    def get_closed_pl(self) -> float:
        """
        Calculate the profit and loss from closed trades.

        Returns:
            float: Profit and loss from closed trades.
        """
        return sum(trade.total_pl() for trade in self.closed_trades)

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
            }
        )

    def plot_performance(self):
        """
        Generate and display performance visualizations.
        """
        if not self.performance_data:
            logger.warning("No performance data available for plotting.")
            return

        df = pd.DataFrame(self.performance_data)
        df.set_index("time", inplace=True)

        # Calculate drawdown
        df["peak"] = df["total_pl"].cummax()
        df["drawdown"] = df["peak"] - df["total_pl"]

        # Create subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, figsize=(12, 24), sharex=True
        )

        # Plot Total P/L
        ax1.plot(df.index, df["total_pl"], label="Total P/L")
        ax1.set_title("Total P/L")
        ax1.set_ylabel("P/L ($)")
        ax1.legend()

        # Plot Closed P/L
        ax2.plot(df.index, df["closed_pl"], label="Closed P/L")
        ax2.set_title("Closed P/L")
        ax2.set_ylabel("P/L ($)")
        ax2.legend()

        # Plot Drawdown
        ax3.fill_between(df.index, df["drawdown"], label="Drawdown")
        ax3.set_title("Drawdown")
        ax3.set_ylabel("Drawdown ($)")
        ax3.legend()

        # Plot Underlying Price
        ax4.plot(df.index, df["underlying_last"], label="Underlying Price")
        ax4.set_title("Underlying Price")
        ax4.set_ylabel("Price ($)")
        ax4.set_xlabel("Date")
        ax4.legend()

        # Plot Active Positions
        ax5.plot(df.index, df["active_positions"], label="Active Positions")
        ax5.set_title("Active Positions")
        ax5.set_ylabel("Positions")
        ax5.set_xlabel("Date")
        ax5.legend()

        plt.tight_layout()
        plt.show()

    def get_closed_trades_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame of closed trades with various attributes.

        Returns:
            pd.DataFrame: DataFrame containing closed trades information.
        """
        closed_trades_data = []
        cumulative_pl = 0

        for trade in self.closed_trades:
            pl = trade.total_pl()
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
            "average_dit": average_dit,
            "average_dit_spread": average_dit_spread,
            "average_exit_dte": average_exit_dte,
        }

        try:
            # Calculate daily P/L changes from performance data
            daily_pl = df["total_pl"].resample("B").last().ffill()
            daily_returns = daily_pl.diff().dropna()
            sharpe_calculator = SharpeRatio()
            metrics["sharpe_ratio"] = sharpe_calculator.calculate(
                daily_returns.values, risk_free_rate=0.02
            )["sharpe_ratio"]
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio: {str(e)}")
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
                df["closed_pl"]
            )["avg_monthly_pl"]
        except Exception as e:
            logger.error(f"Error calculating Average Monthly P/L: {str(e)}")
        try:
            metrics["avg_monthly_pl_nonzero"] = monthly_return_calculator.calculate(
                df["closed_pl"], non_zero_only=True
            )["avg_monthly_pl"]
        except Exception as e:
            logger.error(
                f"Error calculating Average Monthly P/L (Non-Zero Months): {str(e)}"
            )
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
            daily_pl = df["total_pl"].resample("B").last().ffill()
            daily_returns = daily_pl.diff().dropna()

            risk_of_ruin_calculator = RiskOfRuin()
            risk_result = risk_of_ruin_calculator.calculate(
                returns=daily_returns.values,
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
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"CAGR: {metrics['cagr']:.2%}")
            print(f"Average Monthly P/L: ${metrics['avg_monthly_pl']:.2f}")
            print(
                f"Average Monthly P/L (Non-Zero Months): ${metrics['avg_monthly_pl_nonzero']:.2f}"
            )
            print(f"Average Return over Risk: {metrics['return_over_risk']:.2%}")
            print(
                f"Probability of Positive Monthly P/L: {metrics['probability_of_positive_monthly_pl']:.2%}"
            )
            print(
                f"Probability of Positive Monthly Closed P/L: {metrics['probability_of_positive_monthly_closed_pl']:.2%}"
            )
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(
                f"Kelly Criterion Recommendation: {self.calculate_kelly_criterion():.2%}"
            )
            print(f"Risk of Ruin: {metrics['risk_of_ruin']:.2%}")
            print(f"Max Drawdown (Dollars): ${metrics['max_drawdown_dollars']:.2f}")
            print(
                f"Max Drawdown (Percentage): {metrics['max_drawdown_percentage']:.2%}"
            )
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
