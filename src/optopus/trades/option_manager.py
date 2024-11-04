import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from .option_spread import OptionStrategy
from datetime import datetime
from loguru import logger
from dataclasses import dataclass
import numpy as np
from scipy.stats import gaussian_kde
import os
from .entry_conditions import EntryConditionChecker, DefaultEntryCondition


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
    entry_condition: EntryConditionChecker = DefaultEntryCondition()

    def get(self, key, default=None):
        """Get an attribute with a default value if it does not exist."""
        return getattr(self, key, default)


class OptionBacktester:

    def __init__(self, config: Config):
        self.config = config
        self.capital = config.initial_capital
        self.allocation = config.initial_capital
        self.available_to_trade = config.initial_capital
        self.active_trades: List[OptionStrategy] = []
        self.closed_trades: List[OptionStrategy] = []
        self.last_update_time: Optional[datetime] = None
        self.trades_entered_today = 0
        self.trades_entered_this_week = 0
        self.performance_data = []

    def update(self, current_time: datetime, option_chain_df: pd.DataFrame) -> None:
        try:
            current_time = pd.to_datetime(current_time)

            # Update all active trades
            trades_to_close = []
            for trade in self.active_trades:
                trade.update(current_time, option_chain_df)
                if trade.status == "CLOSED":
                    trades_to_close.append(trade)

            # Move closed trades
            if trades_to_close:
                self.active_trades = [
                    t for t in self.active_trades if t.status != "CLOSED"
                ]
                self.closed_trades.extend(trades_to_close)
                pl_change = sum(trade.total_pl() for trade in trades_to_close)
                recovered_capital = sum(
                    trade.get_required_capital() for trade in trades_to_close
                )
                self.capital += pl_change
                self.available_to_trade += recovered_capital
                assert (
                    np.isnan(self.capital) == False
                ), f"Capital is NaN: {self.capital} at {current_time}"

                # Update allocation if gain_reinvesting is True
                if self.config.gain_reinvesting:
                    new_allocation = max(self.capital, self.allocation)
                    added_allocation = new_allocation - self.allocation
                    self.allocation = new_allocation
                    self.available_to_trade += added_allocation

            self._update_trade_counts()
            self.last_update_time = current_time

            # Record performance data after update
            self._record_performance_data(current_time, option_chain_df)
        except Exception as e:
            logger.error(f"Error updating backtester: {str(e)}")

    def add_spread(self, new_spread: OptionStrategy) -> bool:
        try:
            if not self._can_add_spread(new_spread):
                return False

            required_capital = new_spread.get_required_capital()
            self.active_trades.append(new_spread)
            self.available_to_trade -= required_capital
            self._update_trade_counts()
            logger.info(f"Added new spread: {new_spread}")
            return True
        except Exception as e:
            logger.error(f"Error adding spread: {str(e)}")
            return False

    def _can_add_spread(self, new_spread: OptionStrategy) -> bool:
        if self.capital <= 0:
            logger.warning("Cannot add spread: no capital left.")
            return False
        max_capital = min(self.allocation * self.config.position_size, self.capital)
        original_contracts = new_spread.contracts
        new_spread.contracts = min(
            original_contracts,
            int(max_capital // new_spread.get_required_capital_per_contract()),
        )

        if new_spread.contracts == 0:
            logger.warning(
                f"Spread requires more capital than allowed by position size. Skipping spread: {new_spread}"
            )
            return False

        if new_spread.contracts != original_contracts:
            logger.info(
                f"Adjusted spread contracts from {original_contracts} to {new_spread.contracts} to fit position size."
            )
            pass

        conditions = [
            ("No conflict", not self._check_conflict(new_spread)),
            (
                "Meets ROR threshold",
                self.config.ror_threshold is None or self._check_ror(new_spread),
            ),
            (
                "Within max positions",
                len(self.active_trades) < self.config.max_positions,
            ),
            (
                "Within max positions per day",
                self.config.max_positions_per_day is None
                or self.trades_entered_today < self.config.max_positions_per_day,
            ),
            (
                "Within max positions per week",
                self.config.max_positions_per_week is None
                or self.trades_entered_this_week < self.config.max_positions_per_week,
            ),
            ("Within max capital", new_spread.get_required_capital() <= max_capital),
            (
                "Sufficient capital",
                new_spread.get_required_capital() <= self.available_to_trade,
            ),
            (
                "Entry condition met",
                self.config.entry_condition.should_enter(),
            ),
        ]

        for condition_name, condition_result in conditions:
            if not condition_result:
                logger.info(
                    f"Cannot add spread: {condition_name} condition not met"
                )
            else:
                logger.debug(f"Spread meets condition: {condition_name}")

        return all(condition for _, condition in conditions)

    def _update_trade_counts(self) -> None:
        self.trades_entered_today = sum(
            1 for trade in self.active_trades if trade.DIT == 0
        )
        self.trades_entered_this_week = sum(
            1 for trade in self.active_trades if trade.DIT < 7
        )

    def _check_conflict(self, new_spread: OptionStrategy) -> bool:
        return any(
            existing_spread.conflicts_with(new_spread)
            for existing_spread in self.active_trades
        )

    def _check_ror(self, spread: OptionStrategy) -> bool:
        return spread.return_over_risk() >= self.config.ror_threshold

    def get_total_pl(self) -> float:
        return sum(
            trade.total_pl() for trade in self.active_trades + self.closed_trades
        )

    def get_closed_pl(self) -> float:
        return sum(trade.total_pl() for trade in self.closed_trades)

    def get_open_positions(self) -> int:
        return len(self.active_trades)

    def get_closed_positions(self) -> int:
        return len(self.closed_trades)

    def _record_performance_data(
        self, current_time: datetime, option_chain_df: pd.DataFrame
    ) -> None:
        total_pl = self.get_total_pl()
        closed_pl = self.get_closed_pl()
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
            }
        )

    def plot_performance(self):
        """Generate performance visualizations."""
        if not self.performance_data:
            logger.warning("No performance data available for plotting.")
            return

        df = pd.DataFrame(self.performance_data)
        df.set_index("time", inplace=True)

        # Calculate drawdown
        df["peak"] = df["total_pl"].cummax()
        df["drawdown"] = df["peak"] - df["total_pl"]

        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

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

        plt.tight_layout()
        plt.show()

    def get_closed_trades_df(self):
        """
        Compute a dataframe of closed trades with various attributes.

        Returns:
            pd.DataFrame: A dataframe containing information about closed trades.
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
                "exit_time": trade.exit_time,
                "contracts": trade.contracts,
                "entry_underlying_last": trade.entry_underlying_last,
                "exit_underlying_last": trade.exit_underlying_last,
                "entry_net_premium": trade.entry_net_premium,
                "exit_net_premium": trade.exit_net_premium,
                "closed_pl": pl,
                "cumulative_pl": cumulative_pl,
                "return_percentage": trade.return_percentage(),
                "return_over_risk": trade.return_over_risk(),
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
                        f"leg{i+1}_exit_price": leg.current_price,
                    }
                )

            closed_trades_data.append(trade_data)

        return pd.DataFrame(closed_trades_data)

    def calculate_performance_metrics(self):
        """Calculate various performance metrics."""
        if not self.performance_data:
            logger.warning("No performance data available for metric calculation.")
            return None

        df = pd.DataFrame(self.performance_data)
        df.set_index("time", inplace=True)

        closed_trades_df = self.get_closed_trades_df()
        try:
            trade_returns_per_allocation = closed_trades_df["closed_pl"] / self.allocation
        except KeyError:
            trade_returns_per_allocation = np.nan
        
        try:
            average_dit = closed_trades_df["dit"].mean()
        except KeyError:
            average_dit = np.nan

        max_drawdown_dollars, max_drawdown_percentage = self._calculate_max_drawdown(df)

        metrics = {
            "sharpe_ratio": None,
            "profit_factor": None,
            "cagr": None,
            "avg_monthly_pl": None,
            "avg_monthly_pl_nonzero": None,
            "win_rate": None,
            "risk_of_ruin": None,
            "probability_of_positive_monthly_pl": None,
            "probability_of_positive_monthly_closed_pl": None,
            "max_drawdown_dollars": max_drawdown_dollars,
            "max_drawdown_percentage": max_drawdown_percentage,
            "average_dit": average_dit,
        }

        try:
            metrics["sharpe_ratio"] = self._calculate_sharpe_ratio(trade_returns_per_allocation)
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio: {str(e)}")
        try:
            metrics["profit_factor"] = self._calculate_profit_factor(trade_returns_per_allocation)
        except Exception as e:
            logger.error(f"Error calculating Profit Factor: {str(e)}")
        try:
            metrics["cagr"] = self._calculate_cagr(df)
        except Exception as e:
            logger.error(f"Error calculating CAGR: {str(e)}")
        try:
            metrics["avg_monthly_pl"] = self._calculate_avg_monthly_pl(df)
        except Exception as e:
            logger.error(f"Error calculating Average Monthly P/L: {str(e)}")
        try:
            metrics["avg_monthly_pl_nonzero"] = self.calculate_avg_monthly_pl_nonzero(df)
        except Exception as e:
            logger.error(f"Error calculating Average Monthly P/L (Non-Zero Months): {str(e)}")
        try:
            metrics["win_rate"] = self._calculate_win_rate()
        except Exception as e:
            logger.error(f"Error calculating Win Rate: {str(e)}")
        try:
            metrics["risk_of_ruin"] = self.monte_carlo_risk_of_ruin(
                closed_trades_df["closed_pl"].values,
                self.config.initial_capital,
                distribution="histogram",
            )
        except Exception as e:
            logger.error(f"Error calculating Risk of Ruin: {str(e)}")
        try:
            metrics["probability_of_positive_monthly_pl"] = self._calculate_probability_of_positive_monthly_pl(df)
        except Exception as e:
            logger.error(f"Error calculating Probability of Positive Monthly P/L: {str(e)}")
        try:
            metrics["probability_of_positive_monthly_closed_pl"] = self._calculate_probability_of_positive_monthly_closed_pl(df)
        except Exception as e:
            logger.error(f"Error calculating Probability of Positive Monthly Closed P/L: {str(e)}")

        return metrics

    def _calculate_sharpe_ratio(self, daily_returns):
        """Calculate Sharpe Ratio."""
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = (
            daily_returns - risk_free_rate / 252
        )  # Assuming 252 trading days
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_profit_factor(self, daily_returns):
        """Calculate Profit Factor."""
        profits = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        return profits / losses if losses != 0 else np.inf

    def _calculate_cagr(self, df):
        """Calculate Compound Annual Growth Rate."""
        start_value = self.config.initial_capital
        end_value = start_value + df["total_pl"].iloc[-1]
        df["time"] = pd.DatetimeIndex(df.index)
        n_years = (df["time"].iloc[-1] - df["time"].iloc[0]).days / 365.25
        try:
            return (end_value / start_value) ** (1 / n_years) - 1
        except ZeroDivisionError:
            return np.nan

    def _calculate_avg_monthly_pl(self, df):
        """Calculate Average Monthly P/L."""
        monthly_pl = df.set_index("time")["total_pl"].resample("M").last().diff()
        return monthly_pl.mean()

    def _calculate_probability_of_positive_monthly_pl(self, df):
        """Calculate the probability of having a positive monthly P/L."""
        monthly_pl = (
            df.set_index("time")["total_pl"].resample("M").last().diff().dropna()
        )
        positive_months = monthly_pl[monthly_pl > 0]
        total_months = monthly_pl[monthly_pl != 0]
        return len(positive_months) / len(total_months) if len(total_months) > 0 else 0

    def _calculate_probability_of_positive_monthly_closed_pl(self, df):
        """Calculate the probability of having a positive monthly closed P/L."""
        monthly_closed_pl = (
            df.set_index("time")["closed_pl"].resample("M").last().diff().dropna()
        )
        positive_months = monthly_closed_pl[monthly_closed_pl > 0]
        total_months = monthly_closed_pl[monthly_closed_pl != 0]
        return len(positive_months) / len(total_months) if len(total_months) > 0 else 0

    def calculate_avg_monthly_pl_nonzero(self, df):
        """Calculate the average monthly P/L, considering only non-zero months."""
        monthly_pl = df.set_index("time")["total_pl"].resample("M").last().diff().dropna()
        non_zero_months = monthly_pl[monthly_pl != 0]
        return non_zero_months.mean() if not non_zero_months.empty else 0

    def _calculate_max_drawdown(self, df):
        """Calculate the maximum drawdown percentage and dollars."""
        df["peak"] = df["total_pl"].cummax()
        df["drawdown"] = df["peak"] - df["total_pl"]
        max_drawdown_dollars = df["drawdown"].max()
        max_drawdown_percentage = max_drawdown_dollars / self.allocation
        return max_drawdown_dollars, max_drawdown_percentage

    def _calculate_win_rate(self):
        """Calculate Win Rate."""
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.won)
        return winning_trades / total_trades if total_trades > 0 else 0

    def monte_carlo_risk_of_ruin(
        self,
        data,
        initial_balance,
        num_simulations=20000,
        num_steps=252,
        drawdown_threshold_pct=0.25,
        distribution="histogram",
    ):
        # Calculate returns based on allocation
        returns = data / initial_balance

        # Calculate the mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Generate random returns for all simulations and steps
        if distribution == "normal":
            random_returns = np.random.normal(
                mean_return, std_return, size=(num_simulations, num_steps)
            )
        elif distribution == "kde":
            # Estimate the density of the returns using KDE
            kde = gaussian_kde(returns, bw_method="scott")
            # Generate a range of possible return values
            x = np.linspace(np.min(returns), np.max(returns), 1000)
            # Draw samples from the KDE
            samples = kde.resample(size=(num_simulations * num_steps))
            random_returns = samples.T  # Shape: (num_simulations, num_steps)
        elif distribution == "histogram":
            random_returns = np.random.choice(
                returns, size=(num_simulations, num_steps)
            )
        else:
            raise ValueError("Unsupported distribution type.")

        # Calculate balance changes
        balance_changes = random_returns * initial_balance

        # Calculate cumulative balance changes
        cumulative_balance_changes = np.cumsum(balance_changes, axis=1)

        # Calculate balances
        balances = initial_balance + cumulative_balance_changes

        # Calculate peak balances
        peak_balances = np.maximum.accumulate(balances, axis=1)

        # Calculate drawdown thresholds
        drawdown_thresholds = peak_balances - drawdown_threshold_pct * initial_balance

        # Check ruin condition
        ruin_mask = balances <= drawdown_thresholds

        # Count ruined simulations
        ruin_count = np.sum(np.any(ruin_mask, axis=1))

        # Calculate Risk of Ruin
        risk_of_ruin = ruin_count / num_simulations

        return risk_of_ruin

    def update_config(self, **kwargs):
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
                setattr(self.config, key, value)
                # Update related attributes that depend on config
                if key == 'initial_capital' or key == 'allocation':
                    new_allocation = value
                    added_allocation = new_allocation - self.allocation
                    self.allocation = value
                    self.available_to_trade += added_allocation
                    self.capital += added_allocation
                elif key == 'position_size':
                    # Validate position size is between 0 and 1
                    if not 0 < value <= 1:
                        logger.warning(f"Invalid position_size {value}. Must be between 0 and 1.")
                        continue
                updated = True
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        return updated

    def print_performance_summary(self):
        """Print a summary of performance metrics."""
        metrics = self.calculate_performance_metrics()
        if metrics:
            print("\nPerformance Summary:")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"CAGR: {metrics['cagr']:.2%}")
            print(f"Average Monthly P/L: ${metrics['avg_monthly_pl']:.2f}")
            print(f"Average Monthly P/L (Non-Zero Months): ${metrics['avg_monthly_pl_nonzero']:.2f}")
            print(
                f"Probability of Positive Monthly P/L: {metrics['probability_of_positive_monthly_pl']:.2%}"
            )
            print(
                f"Probability of Positive Monthly Closed P/L: {metrics['probability_of_positive_monthly_closed_pl']:.2%}"
            )
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Risk of Ruin: {metrics['risk_of_ruin']:.2%}")
            print(f"Max Drawdown (Dollars): ${metrics['max_drawdown_dollars']:.2f}")
            print(
                f"Max Drawdown (Percentage): {metrics['max_drawdown_percentage']:.2%}"
            )
            print(f"Average Days in Trade: {metrics['average_dit']:.2f}")
        else:
            print("No performance data available for summary.")


if __name__ == "__main__":
    # Load test data
    entry_df = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar/SPY_2024-09-06 15-30.parquet"
    )
    update_df = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar/SPY_2024-09-06 15-45.parquet"
    )
    update_df2 = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-09 09-45.parquet"
    )

    print("\nRunning OptionBacktester tests:")

    # Test 1: Initialization
    print("\nTest 1: Initialization")
    config = Config(
        initial_capital=10000,
        max_positions=5,
        max_positions_per_day=2,
        max_positions_per_week=5,
        position_size=0.1,
        ror_threshold=0.05,
        gain_reinvesting=False,  # Set this to True to test reinvesting gains
    )
    backtester = OptionBacktester(config)
    assert backtester.capital == 10000, "Initial capital not set correctly"
    assert backtester.config.max_positions == 5, "Max positions not set correctly"
    print("Test 1 passed: OptionBacktester initialized correctly")

    # Test 2: Adding a spread
    print("\nTest 2: Adding a spread")
    spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert backtester.add_spread(spread), "Failed to add valid spread"
    assert len(backtester.active_trades) == 1, "Active trades not updated correctly"
    print("Test 2 passed: Spread added successfully")

    # Test 3: Adding conflicting spread
    print("\nTest 3: Adding conflicting spread")
    conflicting_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    assert not backtester.add_spread(conflicting_spread), "Added conflicting spread"
    assert len(backtester.active_trades) == 1, "Active trades incorrectly updated"
    print("Test 3 passed: Conflicting spread not added")

    # Test 4: Updating backtester
    print("\nTest 4: Updating backtester")
    initial_capital = backtester.allocation
    print(f"Initial capital: {initial_capital}")
    initial_pl = backtester.active_trades[0].total_pl()
    print(f"Initial P/L of the trade: {initial_pl}")
    backtester.update("2024-09-06 15:45:00", update_df)
    final_pl = backtester.active_trades[0].total_pl()
    print(f"Final P/L of the trade: {final_pl}")
    print(f"Capital after update: {backtester.capital}")
    assert (
        backtester.trades_entered_today == 1
    ), "Trades entered today not updated correctly"
    assert (
        backtester.trades_entered_this_week == 1
    ), "Trades entered this week not updated correctly"

    assert (
        backtester.capital + backtester.active_trades[0].get_required_capital()
        == initial_capital
    ), f"Capital should reduce by the required capital of the trade. Initial: {initial_capital}, After: {backtester.capital}"
    print("Test 4 passed: Backtester updated successfully")

    # Test 5: Update with position closing
    print("\nTest 5: Update with position closing. Reinitialize backtester.")
    backtester = OptionBacktester(config)
    assert backtester.capital == 10000, "Initial capital not set correctly"
    assert backtester.config.max_positions == 5, "Max positions not set correctly"
    spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
        profit_target=1,
    )
    assert backtester.add_spread(spread), "Failed to add valid spread"
    assert len(backtester.active_trades) == 1, "Active trades not updated correctly"
    backtester.update("2024-09-06 15:45:00", update_df)
    assert len(backtester.active_trades) == 0, "Active trades not closed correctly"
    print("Test 5 passed: Position closed successfully")
    print(f"Capital after update: {backtester.capital}")
    print(f"Closed trades: {backtester.closed_trades}")
    print(f"Active trades: {backtester.active_trades}")
    assert (
        backtester.capital == initial_capital + spread.total_pl()
    ), f"Capital not updated correctly. Initial: {initial_capital}, PL: {initial_pl}, After: {backtester.capital}"

    # Test 6: Maximum positions
    print("\nTest 6: Maximum positions. Reinitialize backtester.")
    backtester = OptionBacktester(config)
    assert backtester.capital == 10000, "Initial capital not set correctly"
    assert backtester.config.max_positions == 5, "Max positions not set correctly"
    assert (
        backtester.config.max_positions_per_day == 2
    ), "Max positions per day not set correctly"
    assert (
        backtester.config.max_positions_per_week == 5
    ), "Max positions per week not set correctly"
    assert backtester.config.position_size == 0.1, "Position size not set correctly"
    assert backtester.config.ror_threshold == 0.05, "ROR threshold not set correctly"

    new_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=540,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    backtester.add_spread(new_spread)
    assert len(backtester.active_trades) == 1, "Active trades not updated correctly"
    assert (
        backtester.trades_entered_today == 1
    ), "Trades entered today not updated correctly"
    assert (
        backtester.trades_entered_this_week == 1
    ), "Trades entered this week not updated correctly"
    new_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=541,
        short_strike=551,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    backtester.add_spread(new_spread)
    assert len(backtester.active_trades) == 2, "Active trades not updated correctly"
    assert (
        backtester.trades_entered_today == 2
    ), "Trades entered today not updated correctly"
    assert (
        backtester.trades_entered_this_week == 2
    ), "Trades entered this week not updated correctly"
    new_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike=543,
        short_strike=553,
        expiration="2024-12-20",
        contracts=1,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    backtester.add_spread(new_spread)
    assert (
        backtester.trades_entered_today == 2
    ), "Trades entered today not updated correctly"
    assert (
        backtester.trades_entered_this_week == 2
    ), "Trades entered this week not updated correctly"

    print(f"Active trades: {backtester.active_trades}")

    assert len(backtester.active_trades) == 2, "Max positions not enforced correctly"
    print("Test 6 passed: Maximum positions enforced correctly")

    # Test 7: Profit/Loss calculation
    print("\nTest 7: Profit/Loss calculation")
    backtester.update("2024-09-06 15:45:00", update_df)
    total_pl = backtester.get_total_pl()
    assert isinstance(total_pl, (float, int)), "Total P/L not calculated as float"
    print(f"Total P/L: {total_pl}")
    closed_pl = backtester.get_closed_pl()
    assert isinstance(closed_pl, (float, int)), "Closed P/L not calculated as float"
    print(f"Closed P/L: {closed_pl}")
    assert closed_pl == 0, "Closed P/L should be 0 because the trade is not closed"
    print("Test 7 passed: Total P/L calculated successfully")

    # Test 8: Adjusting contracts to fit position size
    print("\nTest 8: Adjusting contracts to fit position size")
    backtester = OptionBacktester(config)
    spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="CALL",
        long_strike=560,
        short_strike=550,
        expiration="2024-12-20",
        contracts=1000,
        entry_time="2024-09-06 15:30:00",
        option_chain_df=entry_df,
    )
    max_capital = backtester.allocation * backtester.config.position_size
    expected_contracts = int(max_capital // spread.get_required_capital_per_contract())
    expected_required_capital = (
        spread.get_required_capital_per_contract() * expected_contracts
    )

    print(f"Max capital: {max_capital}")
    print(
        f"Required capital per contract: {spread.get_required_capital_per_contract()}"
    )
    print(f"Expected contracts: {expected_contracts}")
    print(f"Expected required capital: {expected_required_capital}")

    assert backtester.add_spread(spread), "Failed to add spread"
    assert (
        spread.contracts == expected_contracts
    ), f"Contracts not adjusted correctly. Expected: {expected_contracts}, Actual: {spread.contracts}"
    assert (
        spread.get_required_capital() == expected_required_capital
    ), f"Required capital not adjusted correctly. Expected: {expected_required_capital}, Actual: {spread.get_required_capital()}"
    # New test: Check total P/L after adjusting contracts
    initial_capital = backtester.capital
    backtester.update("2024-09-06 15:45:00", update_df)
    total_pl = backtester.get_total_pl()
    expected_pl = spread.total_pl()
    assert (
        abs(total_pl - expected_pl) < 0.01
    ), f"Total P/L not calculated correctly after adjusting contracts. Expected: {expected_pl}, Actual: {total_pl}"

    print(
        "Test 8 passed: Contracts adjusted correctly to fit position size and P/L calculated correctly"
    )

    print("\n--- Performance Visualization Test ---")

    # Create a backtester with some sample data
    config = Config(
        initial_capital=10000,
        max_positions=5,
        max_positions_per_day=2,
        max_positions_per_week=5,
        position_size=0.1,
        ror_threshold=0.05,
    )
    backtester = OptionBacktester(config)

    # Add some sample trades
    for i in range(10):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560 + i,
            short_strike=550 + i,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=entry_df,
        )

        backtester.add_spread(spread)

        # Simulate some updates and closures
        backtester.update("2024-09-06 15:45:00", update_df)
        if i % 2 == 0:
            spread.close_strategy("2024-09-09 09:45:00", update_df2)
            backtester.update("2024-09-09 09:45:00", update_df2)

    # Generate and display the performance plots
    backtester.plot_performance()

    print("Performance visualization test completed")

    # Test 9: Get closed trades dataframe
    print("\nTest 9: Get closed trades dataframe")
    closed_trades_df = backtester.get_closed_trades_df()
    print(closed_trades_df)
    closed_trades_df.to_csv("closed_trades.csv", index=False)
    print("Test 9 passed: Closed trades dataframe generated successfully")

    print("\n--- Performance Metrics Test ---")

    # Create a backtester with some sample data
    config = Config(
        initial_capital=10000,
        max_positions=5,
        max_positions_per_day=2,
        max_positions_per_week=5,
        position_size=0.1,
        ror_threshold=0.05,
    )
    backtester = OptionBacktester(config)

    # Add some sample trades and updates (similar to previous test)
    for i in range(10):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560 + i,
            short_strike=550 + i,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=entry_df,
        )

        backtester.add_spread(spread)

        backtester.update("2024-09-06 15:45:00", update_df)
        if i % 2 == 0:
            spread.close_strategy("2024-09-09 09:45:00", update_df2)
            backtester.update("2024-09-09 09:45:00", update_df2)

    # Print performance summary
    backtester.print_performance_summary()

    print("Performance metrics test completed")

    print("\nAll OptionBacktester tests completed!")
