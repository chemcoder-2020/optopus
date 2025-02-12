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
from optopus.trades.entry_conditions import DefaultEntryCondition
from optopus.trades.external_entry_conditions import (
    EntryOnForecast,
    EntryOnForecastPlusKellyCriterion,
)
import configparser
from exit_condition import ExitCondition

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


# get params function
def get_params(section):
    params = {}
    for option in config[section]._options():
        if "delta" in option or "strike" in option:
            param = config.get(section, option, raw=True)
        elif "exit_time_before_expiration" in option:
            param = config.get(section, option)
            param = pd.Timedelta(param)
            params[option] = param
            continue
        elif "allowed_times" in option:
            param = config.get(section, option)
            param = eval(param)
            params[option] = param
            continue
        else:
            param = config.get(section, option)

        if "." in param:
            try:
                param = float(param)
            except ValueError:
                pass
        elif param.isnumeric():
            param = int(param)
        elif param.lower() in ["true", "false"]:
            param = param.lower() == "true"
        elif "(" in param and ")" in param:
            param = eval(param)
        params[option] = param
    return params


# Strategy Parameters

basic_strategy_params = get_params("STRATEGY_PARAMS")

exit_condition_params = get_params("EXIT_CONDITION")

STRATEGY_PARAMS = basic_strategy_params.copy()
STRATEGY_PARAMS["exit_scheme"] = {
    "class": eval(exit_condition_params["class"]),
}
exit_condition_params.pop("class")
STRATEGY_PARAMS["exit_scheme"]["params"] = exit_condition_params

entry_params = get_params("ENTRY_CONDITION")

external_entry_params = get_params("EXTERNAL_ENTRY_CONDITION")

bt_config = get_params("BACKTESTER_CONFIG")


# Backtester Configuration
BACKTESTER_CONFIG = Config(**bt_config)
BACKTESTER_CONFIG.entry_condition = {
    "class": eval(
        config.get("ENTRY_CONDITION", "class", fallback="DefaultEntryCondition")
    ),
    "params": entry_params,
}
BACKTESTER_CONFIG.external_entry_condition = {
    "class": eval(
        config.get("EXTERNAL_ENTRY_CONDITION", "class", fallback="EntryOnForecast")
    ),
    "params": external_entry_params,
}
BACKTESTER_CONFIG._initialize_entry_condition()
BACKTESTER_CONFIG._initialize_external_entry_condition()
BACKTESTER_CONFIG.trade_type = "Iron Butterfly"
