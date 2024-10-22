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


class Backtest:
    def __init__(
        cls,
        config,
        entry_signal_file,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        strategy_params,
        debug=False,
    ):
        cls.config = config
        cls.entry_signal_file = entry_signal_file
        cls.data_folder = data_folder
        cls.start_date = start_date
        cls.end_date = end_date
        cls.trading_start_time = trading_start_time
        cls.trading_end_time = trading_end_time
        cls.debug = debug
        cls.backtester = OptionBacktester(cls.config)
        cls.strategy_params = strategy_params
        cls.symbol = cls.strategy_params["symbol"]

    @classmethod
    def run_backtest(
        cls, start_date=None, end_date=None, skip_fridays=False, plot_performance=True, backtester=None,
    ):

        if backtester is None:
            backtester = cls.backtester
        if start_date is None:
            start_date = cls.start_date
        if end_date is None:
            end_date = cls.end_date

        # Convert start_date and end_date to strings if they are datetime objects
        if isinstance(start_date, (pd.Timestamp, datetime)):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, (pd.Timestamp, datetime)):
            end_date = end_date.strftime("%Y-%m-%d")
        # Generate time range for trading hours
        time_range = pd.date_range(
            start=f"{start_date} {cls.trading_start_time}",
            end=f"{end_date} {cls.trading_end_time}",
            freq="15min",
        )

        # Filter time range to only include trading hours
        trading_times = time_range.indexer_between_time(
            cls.trading_start_time, cls.trading_end_time
        )
        time_range = time_range[trading_times]

        # Load entry signals
        entry_data = cls.entry_signal_file
        inp = pd.read_csv(entry_data)
        inp["date"] = pd.DatetimeIndex(inp["date"])
        inp["isEntry"] = False
        inp.loc[inp.query(cls.strategy_params["condition"]).index, "isEntry"] = True
        inp["isEntry"] = inp["isEntry"].astype(bool)
        inp.set_index("date", inplace=True)
        inp = inp[["isEntry"]]

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
            filename = f"{cls.symbol}_{time.strftime('%Y-%m-%d %H-%M')}.parquet"
            file_path = os.path.join(cls.data_folder, filename)

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
                if cls.debug:
                    logger.warning(f"Data is empty for {time}. Skipping this update.")
                continue

            if skip_fridays:
                if time.weekday() == 4:
                    continue

            try:
                entry_signal = inp.loc[time, "isEntry"]
            except Exception as e:
                if cls.debug:
                    logger.error(f"Error getting entry signal: {e} at {time}")
                continue

            # Create spread
            try:
                if entry_signal:
                    new_spread = OptionStrategy.create_vertical_spread(
                        symbol=cls.symbol,
                        option_type=cls.strategy_params["option_type"],
                        long_strike=cls.strategy_params["long_delta"],
                        short_strike=cls.strategy_params["short_delta"],
                        expiration=cls.strategy_params["dte"],
                        contracts=cls.strategy_params["contracts"],
                        entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        option_chain_df=option_chain_df,
                        profit_target=cls.strategy_params["profit_target"],
                        stop_loss=cls.strategy_params["stop_loss"],
                        commission=cls.strategy_params["commission"],
                    )
            except Exception as e:
                if cls.debug:
                    logger.error(f"Error creating new spread: {e} at {time}")
                continue

            try:
                if new_spread is not None:
                    if not np.isnan(new_spread.get_required_capital()):
                        if backtester.add_spread(new_spread):
                            if cls.debug:
                                logger.info(f"  Added new spread at {time}")
                    else:
                        if cls.debug:
                            logger.info(
                                f"{time} Spread not added due to NaN required capital."
                            )
            except Exception as e:
                if cls.debug:
                    logger.error(f"Error adding new spread: {e} at {time}")
                continue

            if (
                prev_active_positions != len(backtester.active_trades)
                or prev_capital != backtester.capital
            ):
                if cls.debug:
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

    @classmethod
    def cross_validate(
        cls,
        n_splits: int,
        years_per_split: float,
    ):
        """Cross-validate the backtest."""
        ts_folds = cls.create_time_ranges(
            cls.start_date,
            cls.end_date,
            n_splits,
            years_per_split,
            cls.trading_start_time,
            cls.trading_end_time,
        )

        def run_backtest_for_timerange(time_range: Tuple[str, str]) -> dict:
            """Run backtest for a specific time range and return performance metrics."""

            start_date, end_date = time_range
            backtester = OptionBacktester(cls.config)

            # Modify run_backtest to accept start_date and end_date parameters
            cls.run_backtest(
                start_date, end_date, skip_fridays=False, plot_performance=False, backtester=backtester
            )

            # Calculate and return performance metrics
            metrics = backtester.calculate_performance_metrics()
            profitable = backtester.get_closed_pl() > 0
            metrics.update({"profitable": profitable})
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
            logger.info(f"\n{metric}:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Median: {stats['median']:.4f}")
            logger.info(f"  Std Dev: {stats['std']:.4f}")
            logger.info(f"  Min: {stats['min']:.4f}")
            logger.info(f"  Max: {stats['max']:.4f}")

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
