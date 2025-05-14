from optopus.backtest.vertical_spread import BacktestVerticalSpread
from optopus.trades.option_manager import Config
from optopus.trades.exit_conditions import (
    DefaultExitCondition,
    TrailingStopCondition,
    TimeBasedCondition,
)
from loguru import logger
import pandas as pd


logger.disable("optopus")

# Configuration
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
)
ENTRY_SIGNAL_FILE = f"~/Downloads/stockdata/SPY-AggEMARSICCI.csv"
START_DATE = "2016-01-04"
END_DATE = "2024-10-25"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"
DEBUG = False
# Strategy parameters for vertical spreads

STRATEGY_PARAMS = {
    "symbol": "SPY",
    "option_type": "PUT",
    "dte": 60,
    "short_delta": "ATM",
    "long_delta": "-1",
    "profit_target": None,
    "stop_loss": None,
    "contracts": 1000,
    "condition": "close > 0",  # and 30 < RSI < 70
    "commission": 0.5,
    "exit_scheme": DefaultExitCondition(
        profit_target=30, exit_time_before_expiration=pd.Timedelta(minutes=15)
    ),
}

BACKTESTER_CONFIG = Config(
    initial_capital=10000,
    max_positions=10,
    max_positions_per_day=1,
    max_positions_per_week=None,
    position_size=0.1,
    ror_threshold=0,
    gain_reinvesting=False,
    verbose=False,
)


backtest = BacktestVerticalSpread(
    config=BACKTESTER_CONFIG,
    entry_signal_file=ENTRY_SIGNAL_FILE,
    data_folder=DATA_FOLDER,
    start_date=START_DATE,
    end_date=END_DATE,
    trading_start_time=TRADING_START_TIME,
    trading_end_time=TRADING_END_TIME,
    strategy_params=STRATEGY_PARAMS,
    debug=DEBUG,
)
cv = backtest.cross_validate(8 * 4, 1)
print("\nCross-Validation Results:")
print("==========================")
for metric, stats in cv.items():
    print(f"\n{metric}:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Std Dev: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
