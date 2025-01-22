import pandas as pd
from optopus.trades.option_manager import Config
from optopus.trades.exit_conditions import (
    DefaultExitCondition,
)
from exit_condition import ExitCondition
from entry_condition import BotEntryCondition, EntryCondition
from optopus.trades.entry_conditions import (
    CapitalRequirementCondition,
    PositionLimitCondition,
    RORThresholdCondition,
    ConflictCondition,
    CompositeEntryCondition,
    DefaultEntryCondition
)
from optopus.trades.external_entry_conditions import EntryOnForecast
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
    "option_type": config.get("STRATEGY_PARAMS", "option_type"),
    "dte": config.getint("STRATEGY_PARAMS", "dte"),
    "short_delta": config.get("STRATEGY_PARAMS", "short_delta", raw=True),
    "long_delta": config.get("STRATEGY_PARAMS", "long_delta", raw=True),
    "contracts": config.getint("STRATEGY_PARAMS", "contracts"),
    "commission": config.getfloat("STRATEGY_PARAMS", "commission", fallback=0.5),
    "exit_scheme": {
        "class": eval(config.get("EXIT_CONDITION", "class")),
        "params": {
            "profit_target": config.getfloat(
                "EXIT_CONDITION", "profit_target", fallback=80
            ),
            "trigger": config.getfloat("EXIT_CONDITION", "trigger", fallback=40),
            "stop_loss": config.getfloat("EXIT_CONDITION", "stop_loss", fallback=15),
            "exit_time_before_expiration": pd.Timedelta(
                config.get(
                    "EXIT_CONDITION",
                    "exit_time_before_expiration",
                    fallback="15 minutes",
                )
            ),
            "window_size": config.getint("EXIT_CONDITION", "window_size", fallback=3),
        },
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
    entry_condition={
        "class": eval(
            config.get("ENTRY_CONDITION", "class", fallback="DefaultEntryCondition")
        ),
        "params": {
            "window_size": config.getint("ENTRY_CONDITION", "window_size", fallback=25),
            "fluctuation": config.getfloat(
                "ENTRY_CONDITION", "fluctuation", fallback=0.15
            ),
            "check_closed_trades": config.getboolean(
                "ENTRY_CONDITION", "check_closed_trades", fallback=True
            ),
            "trailing_entry_direction": config.get(
                "ENTRY_CONDITION", "trailing_entry_direction", fallback="bullish"
            ),
            "trailing_entry_threshold": config.getfloat(
                "ENTRY_CONDITION", "trailing_entry_threshold", fallback=0.2
            ),
            "method": config.get("ENTRY_CONDITION", "method", fallback="percent"),
            "trailing_entry_reset_period": config.get(
                "ENTRY_CONDITION", "trailing_entry_reset_period", fallback=None
            ),
        },
    },
    external_entry_condition={
        "class": eval(
            config.get("EXTERNAL_ENTRY_CONDITION", "class", fallback="EntryOnForecast")
        ),
        "params": {
            "ohlc": config.get("EXTERNAL_ENTRY_CONDITION", "ohlc"),
            "atr_period": config.getint(
                "EXTERNAL_ENTRY_CONDITION", "atr_period", fallback=14
            ),
            "linear_regression_lag": config.getint(
                "EXTERNAL_ENTRY_CONDITION", "linear_regression_lag", fallback=14
            ),
            "median_trend_short_lag": config.getint(
                "EXTERNAL_ENTRY_CONDITION", "median_trend_short_lag", fallback=50
            ),
            "median_trend_long_lag": config.getint(
                "EXTERNAL_ENTRY_CONDITION", "median_trend_long_lag", fallback=200
            ),
        },
    },
    trade_type=config.get(
        "BACKTESTER_CONFIG", "trade_type", fallback="Vertical Spread"
    ),
)
