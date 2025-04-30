from optopus.backtest import BaseBacktest
from loguru import logger

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

logger.disable("optopus")

backtest = BaseBacktest(
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
