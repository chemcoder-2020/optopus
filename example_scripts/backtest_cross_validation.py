import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
import logging
from typing import List, Tuple

from option_manager import OptionBacktester, BacktesterConfig
import contextlib
import joblib
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.disabled = True


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


def create_time_ranges(
    start_date: str, end_date: str, n_splits: int, years_per_split: float
) -> List[Tuple[str, str]]:
    """Create time ranges for cross-validation."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="15min")

    full_range = pd.Series(full_range).loc[
        full_range.indexer_between_time("09:45", "15:45")
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


def cross_validate_backtest(
    n_splits: int = 100, years_per_split: int = 3, which_bot: str = "45DTE"
) -> dict:
    """Perform cross-validation of the backtest."""
    if which_bot == "45DTE":
        from backtest_verticals import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )

    elif which_bot == "SPS4DTE":
        from backtest_sps_4dte import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )

    elif which_bot == "ABDUCTOR":
        from backtest_premium_abductor import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )

    elif which_bot == "THEEDGE":
        from backtest_TheEdge import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )
    
    elif which_bot == "1DTE":
        from backtest_1dte import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )
    
    elif which_bot == "nDTE":
        from backtest_sps_nDTE import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )
    elif which_bot == "Naked Call":
        from backtest_naked_calls import (
            run_backtest,
            SYMBOL,
            START_DATE,
            END_DATE,
            TRADING_START_TIME,
            TRADING_END_TIME,
            BACKTESTER_CONFIG,
        )
    else:
        raise ValueError(f"Invalid bot name: {which_bot}")

    def run_backtest_for_timerange(
        time_range: Tuple[str, str]
    ) -> dict:
        """Run backtest for a specific time range and return performance metrics."""

        start_date, end_date = time_range
        backtester = OptionBacktester(BACKTESTER_CONFIG)

        # Modify run_backtest to accept start_date and end_date parameters
        run_backtest(backtester, start_date, end_date, plot_performance=False)

        # Calculate and return performance metrics
        metrics = backtester.calculate_performance_metrics()
        profitable = backtester.get_closed_pl() > 0
        metrics.update({"profitable": profitable})
        return metrics

    time_ranges = create_time_ranges(START_DATE, END_DATE, n_splits, years_per_split)

    # Run backtests in parallel
    with tqdm_joblib(tqdm(desc="Backtest Validation", total=n_splits)) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(run_backtest_for_timerange)(tr) for tr in time_ranges
        )

    # Aggregate results
    aggregated_results = {}
    for metric in results[0].keys():
        values = [result[metric] for result in results]
        aggregated_results[metric] = {
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
        }

    return aggregated_results


def print_cross_validation_results(results: dict):
    """Print the cross-validation results."""
    print("\nCross-Validation Results:")
    print("==========================")
    for metric, stats in results.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")


if __name__ == "__main__":
    logger.info("Starting cross-validation...")
    which_bot = "nDTE"  # SPS4DTE, ABDUCTOR, THEEDGE, 1DTE, 45DTE, Naked Call, nDTE
    results = cross_validate_backtest(n_splits=9*4, years_per_split=1.5, which_bot=which_bot)
    print_cross_validation_results(results)
    logger.info("Cross-validation completed.")
