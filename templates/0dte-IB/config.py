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
)

# Configuration
ohlc_file = "/Users/traderHuy/Documents/AITrading/TOS Bot 2/bigcaps_data/alpaca_SPY_15m_rebase.csv"
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
)
START_DATE = "2024-01-03"
END_DATE = "2024-12-31"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"
DEBUG = False

[STRATEGY_PARAMS]
symbol = "SPY"
dte = 45
strike = "ATM"
put_long_strike = "ATM-1%"
put_short_strike = "ATM"
call_short_strike = "ATM+2%"
call_long_strike = "ATM+3%"
contracts = 1000
commission = 0.5
exit_scheme = DefaultExitCondition(
    profit_target=50,
    exit_time_before_expiration=pd.Timedelta("15 minutes"),
    window_size=5,
)

[BACKTESTER_CONFIG]
initial_capital = 20000
max_positions = 10
max_positions_per_day = 1
max_positions_per_week = None
position_size = 0.1
ror_threshold = 0.2
gain_reinvesting = False
verbose = False
entry_condition = CompositeEntryCondition(
    [
        CapitalRequirementCondition(),
        PositionLimitCondition(),
        RORThresholdCondition(),
        ConflictCondition(check_closed_trades=True),
    ]
)
