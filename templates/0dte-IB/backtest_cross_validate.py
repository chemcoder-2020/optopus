from optopus.backtest.naked_call import BacktestNakedCall
from loguru import logger
from config import (
    DATA_FOLDER,
    START_DATE,
    END_DATE,
    TRADING_START_TIME,
    TRADING_END_TIME,
    DEBUG,
    STRATEGY_PARAMS,
    BACKTESTER_CONFIG,
)
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
basename = os.path.split(dname)[0]

logger.add(
    f"{cwd}/45dteCrossValidate.log",
    rotation="10 MB",
    retention="60 days",
    compression="zip",
)

logger.disable("optopus")

backtest = BacktestNakedCall(
    config=BACKTESTER_CONFIG,
    data_folder=DATA_FOLDER,
    start_date=START_DATE,
    end_date=END_DATE,
    trading_start_time=TRADING_START_TIME,
    trading_end_time=TRADING_END_TIME,
    strategy_params=STRATEGY_PARAMS,
    debug=DEBUG,
)
logger.info("==================================================")
logger.info(f"Strategy Parameters: {STRATEGY_PARAMS}")
cv = backtest.cross_validate(20, 1.1)
logger.info("\nCross-Validation Results:")
logger.info("==========================")
for metric, stats in cv.items():
    if "true_ratio" in stats:
        logger.info(f"\n{metric}:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  True Ratio: {stats['true_ratio']:.4f}")
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Count: {stats['count']}")
        print(f"  True Ratio: {stats['true_ratio']:.4f}")
    else:
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
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  25th Percentile: {stats['percentile_25']:.4f}")
        print(f"  75th Percentile: {stats['percentile_75']:.4f}")
        print(f"  90th Percentile: {stats['percentile_90']:.4f}")
        print(f"  95th Percentile: {stats['percentile_95']:.4f}")
        print(f"  IQR: {stats['iqr']:.4f}")
        print(f"  Skewness: {stats['skewness']:.4f}")
        print(f"  Kurtosis: {stats['kurtosis']:.4f}")
        print(f"  Sample Count: {stats['count']}")
