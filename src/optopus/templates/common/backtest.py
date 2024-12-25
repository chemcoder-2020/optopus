from strategy_selection import BacktestStrategy
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
bt = backtest.run_backtest()
closed_trades_df = bt.get_closed_trades_df().set_index("exit_time").sort_index()
