from strategy_selection import BacktestStrategy
from loguru import logger
import os
from optopus.utils.config_parser import IniConfigParser

parser = IniConfigParser("config.ini")
config = parser.get_config()  # Returns Config dataclass instance
strategy_params = parser.get_strategy_params()  # Returns dict of strategy parameters
general_params = parser.get_general_params()

DATA_FOLDER = general_params["data_folder"]
START_DATE = general_params["start_date"]
END_DATE = general_params["end_date"]
TRADING_START_TIME = general_params["trading_start_time"]
TRADING_END_TIME = general_params["trading_end_time"]
DEBUG = general_params["debug"]
STRATEGY_PARAMS = strategy_params
BACKTESTER_CONFIG = config


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
basename = os.path.split(dname)[0]

logger.add(
    f"{cwd}/CrossValidate.log",
    rotation="10 MB",
    retention="60 days",
    compression="zip",
)

logger.disable("optopus")

backtest = BacktestStrategy(
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
for metric, stats in cv['aggregated'].items():
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
