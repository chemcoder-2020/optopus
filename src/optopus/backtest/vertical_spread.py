import pandas as pd
import os
from ..trades.option_manager import OptionBacktester
from ..trades.option_spread import OptionStrategy
import numpy as np
from loguru import logger
from datetime import datetime
from typing import List, Tuple
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class BacktestVerticalSpread:
    def __init__(
        self,
        config,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        strategy_params,
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
        self.strategy_params = strategy_params
        self.symbol = self.strategy_params["symbol"]

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
                    logger.warning(f"No data available for {time}. Skipping this update.")
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

            # Create spread
            try:
                new_spread = OptionStrategy.create_vertical_spread(
                    symbol=self.symbol,
                    option_type=self.strategy_params["option_type"],
                    long_strike=self.strategy_params["long_delta"],
                    short_strike=self.strategy_params["short_delta"],
                    expiration=self.strategy_params["dte"],
                    contracts=self.strategy_params["contracts"],
                    entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    option_chain_df=option_chain_df,
                    profit_target=self.strategy_params.get("profit_target"),
                    stop_loss=self.strategy_params.get("stop_loss"),
                    commission=self.strategy_params.get("commission", 0),
                    exit_scheme=self.strategy_params.get("exit_scheme"),
                )
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

        print("\nBacktest completed!")
        print(f"Final capital: ${backtester.capital:.2f}")
        print(f"Total P&L: ${backtester.get_total_pl():.2f}")
        print(f"Closed P&L: ${backtester.get_closed_pl():.2f}")
        print(f"Number of closed positions: {backtester.get_closed_positions()}")

        if plot_performance:
            backtester.plot_performance()
        backtester.print_performance_summary()
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
    ):
        """Cross-validate the backtest."""
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
            results = Parallel(n_jobs=-1)(
                delayed(run_backtest_for_timerange)(tr) for tr in ts_folds
            )

        # Aggregate results
        aggregated_results = {}
        for metric in results[0].keys():
            if metric != "performance_data":
                values = [result[metric] for result in results]
                aggregated_results[metric] = {
                    "mean": np.nanmean(values),
                    "median": np.nanmedian(values),
                    "std": np.nanstd(values),
                    "min": np.nanmin(values),
                    "max": np.nanmax(values),
                }

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

        # Plot and save closed_pl vs timedelta for each split
        for i, result in enumerate(results):
            performance_data = result["performance_data"]
            df = pd.DataFrame(performance_data, columns=["time", "closed_pl"])
            df["time"] = pd.to_datetime(df["time"])
            df["timedelta"] = df["time"] - df["time"].iloc[0]

            import matplotlib.ticker as ticker

            def timeTicks(x, pos):
                d = datetime.timedelta(seconds=x)
                return str(d)

            formatter = ticker.FuncFormatter(timeTicks)

            plt.figure(figsize=(10, 6))
            plt.plot(df["timedelta"].dt.total_seconds(), df["closed_pl"], marker='o', linestyle='-')
            plt.title(f"Closed P&L vs Time for Split {i + 1}")
            plt.xlabel("Time (timedelta)")
            plt.ylabel("Closed P&L")
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(formatter)
            # plt.tight_layout()
            plt.savefig(f"closed_pl_vs_time_split_{i + 1}.png")
            plt.close()

        return aggregated_results


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
