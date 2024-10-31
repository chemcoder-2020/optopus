from optopus.backtest.vertical_spread import BacktestVerticalSpread
from optopus.trades.option_manager import Config
from optopus.trades.exit_conditions import DefaultExitCondition, TrailingStopCondition, TimeBasedCondition, CompositeExitCondition
from loguru import logger
import pandas as pd

logger.disable("optopus")

# Configuration
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
)
ENTRY_SIGNAL_FILE = f"~/Downloads/stockdata/SPY-AggEMARSICCI.csv"
START_DATE = "2023-01-04"
END_DATE = "2024-01-04"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"
DEBUG = False

# Strategy parameters for vertical spreads
exit_condition = CompositeExitCondition([TrailingStopCondition(trigger=40, stop_loss=15), TimeBasedCondition(exit_time_before_expiration=pd.Timedelta(minutes=15))], logical_operation="OR")

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
    # "exit_scheme": DefaultExitCondition(profit_target=20, exit_time_before_expiration=pd.Timedelta(minutes=15)),
    "exit_scheme": exit_condition
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
    debug=DEBUG
)
bt = backtest.run_backtest()
closed_trades_df = bt.get_closed_trades_df()
closed_trades_df.iloc[-1]
bt.closed_trades[80]
