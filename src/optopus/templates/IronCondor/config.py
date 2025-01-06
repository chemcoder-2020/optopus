import pandas as pd
from optopus.trades.option_manager import Config
from optopus.trades.exit_conditions import (
    DefaultExitCondition,
)
from optopus.trades.entry_conditions import (
    CapitalRequirementCondition,
    PositionLimitCondition,
    RORThresholdCondition,
    ConflictCondition,
    CompositeEntryCondition,
)
import configparser

# Read configuration from config.txt
config = configparser.ConfigParser()
config.read("config.ini")

# General Configuration
ohlc_file = config.get("GENERAL", "ohlc_file")
DATA_FOLDER = config.get("GENERAL", "DATA_FOLDER")
START_DATE = config.get("GENERAL", "START_DATE")
END_DATE = config.get("GENERAL", "END_DATE")
TRADING_START_TIME = config.get("GENERAL", "TRADING_START_TIME")
TRADING_END_TIME = config.get("GENERAL", "TRADING_END_TIME")
DEBUG = config.getboolean("GENERAL", "DEBUG")

# Strategy Parameters
STRATEGY_PARAMS = {
    "symbol": config.get("STRATEGY_PARAMS", "symbol"),
    "put_long_strike": config.get("STRATEGY_PARAMS", "put_long_strike", raw=True),
    "put_short_strike": config.get("STRATEGY_PARAMS", "put_short_strike", raw=True),
    "call_short_strike": config.get("STRATEGY_PARAMS", "call_short_strike", raw=True),
    "call_long_strike": config.get("STRATEGY_PARAMS", "call_long_strike", raw=True),
    "dte": config.getint("STRATEGY_PARAMS", "dte"),
    "contracts": config.getint("STRATEGY_PARAMS", "contracts"),
    "commission": config.getfloat("STRATEGY_PARAMS", "commission", fallback=0.5),
    "exit_scheme": {
        "class": eval(config.get("EXIT_CONDITION", "class")),
        "params": {
            "profit_target": config.getfloat("EXIT_CONDITION", "profit_target", fallback=50),
            "exit_time_before_expiration": pd.Timedelta(
                config.get(
                    "EXIT_CONDITION", "exit_time_before_expiration", fallback="15 minutes"
                )
            ),
            "window_size": config.getint("EXIT_CONDITION", "window_size", fallback=5),
        }
    },
}

# Backtester Configuration
BACKTESTER_CONFIG = Config(
    initial_capital=config.getfloat("BACKTESTER_CONFIG", "initial_capital"),
    max_positions=config.getint("BACKTESTER_CONFIG", "max_positions", fallback=10),
    max_positions_per_day=config.getint(
        "BACKTESTER_CONFIG", "max_positions_per_day", fallback=1
    ),
    max_positions_per_week=config.getint(
        "BACKTESTER_CONFIG", "max_positions_per_week", fallback=1000000000000
    ),
    position_size=config.getfloat("BACKTESTER_CONFIG", "position_size", fallback=0.05),
    ror_threshold=config.getfloat("BACKTESTER_CONFIG", "ror_threshold", fallback=0),
    gain_reinvesting=config.getboolean(
        "BACKTESTER_CONFIG", "gain_reinvesting", fallback=False
    ),
    verbose=config.getboolean("BACKTESTER_CONFIG", "verbose", fallback=False),
    entry_condition=CompositeEntryCondition(
        [
            CapitalRequirementCondition(),
            PositionLimitCondition(),
            RORThresholdCondition(),
            ConflictCondition(
                check_closed_trades=config.getboolean(
                    "BACKTESTER_CONFIG", "check_closed_trades", fallback=True
                )
            ),
        ]
    ),
    trade_type=config.get("BACKTESTER_CONFIG", "trade_type", fallback="Iron Condor"),
)
