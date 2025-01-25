import pandas as pd
import os
from ..trades.option_manager import OptionBacktester
from ..trades.entry_conditions import PositionLimitCondition
import numpy as np
import scipy.stats
from loguru import logger
from datetime import datetime
from typing import List, Tuple
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseBacktest(ABC):
    def __init__(
        self,
        config,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        debug=False,
    ):
        self.config = config
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.trading_start_time = trading_start_time
        self.trading_end_time = trading_end_time
        self.debug = debug
        self.backtester = OptionBacktester(self.config)

    @abstractmethod
    def create_spread(self, time, option_chain_df):
        pass

    def run_backtest(
        self,
        start_date=None,
        end_date=None,
        skip_fridays=False,
        plot_performance=True,
        backtester=None,
    ):

        if backtester is None:
            backtester = self.backtester
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        # Convert start_date and end_date to strings if they are datetime objects
        if isinstance(start_date, (pd.Timestamp, datetime)):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, (pd.Timestamp, datetime)):
            end_date = end_date.strftime("%Y-%m-%d")

        # Generate time range for trading hours
        time_range = pd.date_range(
            start=f"{start_date} {self.trading_start_time}",
            end=f"{end_date} {self.trading_end_time}",
            freq="15min",
        )

        # Filter time range to only include trading hours
        trading_times = time_range.indexer_between_time(
            self.trading_start_time, self.trading_end_time
        )
        time_range = time_range[trading_times]

        prev_active_positions = None
        prev_capital = None
        prev_year = None

        for i, time in enumerate(time_range):
            new_spread = None
            option_chain_df = None
            year = time.year

            if year != prev_year:
                logger.info(f"Processing year: {year}")
                prev_year = year

            filename = f"{self.symbol}_{time.strftime('%Y-%m-%d %H-%M')}.parquet"
            file_path = os.path.join(self.data_folder, filename)

            if not os.path.exists(file_path):
                if self.debug:
                    logger.warning(
                        f"No data available for {time}. Skipping this update."
                    )
                continue

            option_chain_df = pd.read_parquet(file_path)
            if not option_chain_df.empty:
                backtester.update(time, option_chain_df)
            else:
                if self.debug:
                    logger.warning(f"Data is empty for {time}. Skipping this update.")
                continue

            if skip_fridays and time.weekday() == 4:
                continue

            if not PositionLimitCondition().should_enter(None, backtester, time):
                continue

            # Create spread
            try:
                new_spread = self.create_spread(time, option_chain_df)
            except Exception as e:
                if self.debug:
                    logger.error(f"Error creating new spread: {e} at {time}")
                continue

            try:
                if new_spread is not None:
                    if not np.isnan(new_spread.get_required_capital()):
                        if backtester.add_spread(new_spread):
                            if self.debug:
                                logger.info(f"  Added new spread at {time}")
                    else:
                        if self.debug:
                            logger.info(
                                f"{time} Spread not added due to NaN required capital."
                            )
            except Exception as e:
                if self.debug:
                    logger.error(f"Error adding new spread: {e} at {time}")
                continue

            if (
                prev_active_positions != len(backtester.active_trades)
                or prev_capital != backtester.capital
            ):
                if self.debug:
                    logger.info(
                        f"  Time: {time}, Active trades: {len(backtester.active_trades)}, Capital: ${backtester.capital:.2f}, PL: ${backtester.get_total_pl():.2f}"
                    )
                prev_active_positions = len(backtester.active_trades)
                prev_capital = backtester.capital

        try:
            print("\nBacktest completed!")
            print(f"Final capital: ${backtester.capital:.2f}")
            print(f"Total P&L: ${backtester.get_total_pl():.2f}")
            print(f"Closed P&L: ${backtester.get_closed_pl():.2f}")
            print(f"Number of closed positions: {backtester.get_closed_positions()}")

            if plot_performance:
                backtester.plot_performance()
            backtester.print_performance_summary()
        except Exception:
            pass
        return backtester

    @classmethod
    def create_time_ranges(
        cls,
        start_date: str,
        end_date: str,
        n_splits: int,
        years_per_split: float,
        trading_start_time: str,
        trading_end_time: str,
    ) -> List[Tuple[str, str]]:
        """Create time ranges for cross-validation."""
        full_range = pd.date_range(start=start_date, end=end_date, freq="15min")

        full_range = pd.Series(full_range).loc[
            full_range.indexer_between_time(trading_start_time, trading_end_time)
        ]
        full_range = full_range[full_range.dt.weekday < 5]
        full_range = full_range.sort_values(ascending=True)
        full_range = full_range.reset_index(drop=True)

        total_days = len(full_range)
        split_size = int(years_per_split * 252 * 26)

        ts_folds = []
        for i in range(n_splits):
            start_idx = int(i * (total_days - split_size) / (n_splits - 1))
            end_idx = start_idx + split_size
            if end_idx > total_days:
                end_idx = total_days
            fold = full_range.iloc[start_idx:end_idx]
            ts_folds.append(
                (
                    fold.iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                    fold.iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        return ts_folds

    def cross_validate(
        self,
        n_splits: int,
        years_per_split: float,
        n_jobs: int = -1,
    ):
        """Cross-validate the backtest."""
        logger.info("\nStarting Cross-Validation")
        logger.info("==========================")
        logger.info("Cross-Validation Parameters:")
        logger.info(f"Number of splits: {n_splits}")
        logger.info(f"Years per split: {years_per_split}")
        logger.info("\nBacktest Parameters:")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(
            f"Trading hours: {self.trading_start_time} to {self.trading_end_time}"
        )
        logger.info("\nStrategy Configuration:")
        logger.info(f"  Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"  Max positions: {self.config.max_positions}")
        logger.info(f"  Max positions per day: {self.config.max_positions_per_day}")
        logger.info(f"  Max positions per week: {self.config.max_positions_per_week}")
        logger.info(f"  Position size: {self.config.position_size:.1%}")
        logger.info(
            f"  ROR threshold: {self.config.ror_threshold:.1%}"
            if self.config.ror_threshold
            else "  ROR threshold: None"
        )
        logger.info(f"  Gain reinvesting: {self.config.gain_reinvesting}")
        logger.info(f"  Trade type: {self.config.trade_type}")
        logger.info(f"  Ticker: {self.config.ticker}")
        logger.info(
            f"  Entry condition: {self.config.entry_condition.__class__.__name__}"
        )
        logger.info("==========================\n")

        ts_folds = self.create_time_ranges(
            self.start_date,
            self.end_date,
            n_splits,
            years_per_split,
            self.trading_start_time,
            self.trading_end_time,
        )

        def run_backtest_for_timerange(time_range: Tuple[str, str]) -> dict:
            """Run backtest for a specific time range and return performance metrics."""

            start_date, end_date = time_range
            backtester = OptionBacktester(self.config)

            # Modify run_backtest to accept start_date and end_date parameters
            self.run_backtest(
                start_date,
                end_date,
                skip_fridays=False,
                plot_performance=False,
                backtester=backtester,
            )

            # Calculate and return performance metrics
            metrics = backtester.calculate_performance_metrics()
            profitable = backtester.get_closed_pl() > 0
            perf_data = backtester.performance_data
            metrics.update({"profitable": profitable})
            metrics.update({"performance_data": perf_data})
            return metrics

        with tqdm_joblib(
            tqdm(desc="Backtest Validation", total=n_splits)
        ) as progress_bar:
            # Store both time range and result together
            results_with_tr = Parallel(n_jobs=n_jobs)(
                delayed(lambda tr: (tr, run_backtest_for_timerange(tr)))(tr) 
                for tr in ts_folds
            )

        # Unpack the results while preserving order
        time_range_results = {tr: result for tr, result in results_with_tr}

        # Create aggregated results from the ordered results
        results = [result for _, result in results_with_tr]
        from ..metrics import Aggregator
        aggregated_results = Aggregator.aggregate(results)

        logger.info("\nCross-Validation Results:")
        logger.info("==========================")
        for metric, stats in aggregated_results.items():
            if metric != "performance_data":
                logger.info(f"\n{metric}:")
                logger.info(f"  Mean: {stats['mean']:.4f}")
                logger.info(f"  Median: {stats['median']:.4f}")
                logger.info(f"  Std Dev: {stats['std']:.4f}")
                logger.info(f"  Min: {stats['min']:.4f}")
                logger.info(f"  Max: {stats['max']:.4f}")
                logger.info(f"  25th Percentile: {stats['percentile_25']:.4f}")
                logger.info(f"  75th Percentile: {stats['percentile_75']:.4f}")
                logger.info(f"  90th Percentile: {stats['percentile_90']:.4f}")
                logger.info(f"  95th Percentile: {stats['percentile_95']:.4f}")
                logger.info(f"  IQR: {stats['iqr']:.4f}")
                logger.info(f"  Skewness: {stats['skewness']:.4f}")
                logger.info(f"  Kurtosis: {stats['kurtosis']:.4f}")
                logger.info(f"  Sample Count: {stats['count']}")

        # Plot and save closed_pl vs timedelta for all splits on one chart
        import matplotlib.ticker as ticker

        plt.figure(figsize=(10, 6))
        alpha = 0.5  # Set a single alpha value for all splits

        for i, result in enumerate(results):
            performance_data = result["performance_data"]
            df = pd.DataFrame(performance_data, columns=["time", "closed_pl"])
            df["time"] = pd.to_datetime(df["time"])
            df["timedelta"] = df["time"] - df["time"].iloc[0]

            plt.plot(
                df["timedelta"].dt.total_seconds(),
                df["closed_pl"],
                linestyle="-",
                ms=0,
                alpha=alpha,
                label=f"Split {i + 1}",
            )

        def timeTicks(x, pos):
            d = pd.Timedelta(seconds=x)
            return str(d)

        formatter = ticker.FuncFormatter(timeTicks)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.grid(True)
        plt.title("Closed P&L vs Time for All Splits")
        plt.xlabel("Time (timedelta)", rotation=45)
        plt.ylabel("Closed P&L")
        plt.tight_layout()
        plt.savefig("closed_pl_vs_time_all_splits.png")
        plt.close()

        # Create dict mapping time ranges to their results
        time_range_results = {
            tr: result for tr, result in zip(ts_folds, results)
        }
        
        return {
            "aggregated": aggregated_results,
            "individual": time_range_results
        }


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
