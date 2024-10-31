import os
import dill
from optopus.trades.trade_manager import TradingManager
from optopus.trades.option_spread import OptionStrategy
from optopus.trades.option_manager import Config
from optopus.trades.exit_conditions import DefaultExitCondition
import pandas as pd
import dotenv
from loguru import logger

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
basename = os.path.split(dname)[0]

# Configure loguru to write logs to a file
logger.add(
    f"{cwd}/60dteBot.log", rotation="10 MB", retention="60 days", compression="zip"
)

pd.options.display.max_columns = 50
# Extra configuration
automation_on = True
management_on = True

# os.getenv("SCHWAB_REDIRECT_URI")
dotenv.load_dotenv(os.path.join(basename, ".env"))

logger.enable("optopus")

config = Config(
    initial_capital=10000,
    max_positions=10,
    max_positions_per_day=1,
    max_positions_per_week=None,
    position_size=0.1,
    ror_threshold=0,
    gain_reinvesting=False,  # Set this to True to test reinvesting gains
    verbose=False,
    ticker="SPY",
    broker="Schwab",
    client_id=os.getenv("SCHWAB_CLIENT_ID"),
    client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
    redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
    token_file=os.getenv("SCHWAB_TOKEN_FILE", "token.json"),
)

# Strategy parameters for vertical spreads
STRATEGY_PARAMS = {
    "symbol": "SPY",
    "option_type": "PUT",
    "dte": 60,
    "short_delta": "ATM",
    "long_delta": "-1",
    "profit_target": 40,
    "stop_loss": None,
    "contracts": 100,
    "condition": True,
    "commission": 0.5,
}

exit_condition = DefaultExitCondition(
    profit_target=40, exit_time_before_expiration=pd.Timedelta(minutes=15)
)

# Initialize the trading manager
if os.path.exists(f"{cwd}/trading_manager60dte.pkl"):
    trading_manager = dill.load(open(f"{cwd}/trading_manager60dte.pkl", "rb"))
else:
    trading_manager = TradingManager(config)

trading_manager.auth_refresh()

bar = pd.Timestamp.now(tz="America/New_York").floor("15min").tz_localize(None)

if trading_manager.active_trades and management_on:
    trading_manager.update_orders()
    logger.info(f"{bar}: Updated orders")
    logger.info(f"Active trades: {len(trading_manager.active_trades)}")

entry_conditions = [
    ("Position Cap", len(trading_manager.active_trades) < config.max_positions),
    ("Technical", STRATEGY_PARAMS["condition"]),
    (
        "Daily Position Cap",
        trading_manager.trades_entered_today < config.max_positions_per_day,
    ),
    ("Automation Switch On", automation_on),
]

can_enter = all([condition[1] for condition in entry_conditions])

if can_enter:
    option_chain_df = trading_manager.option_broker.data.get_option_chain(config.ticker)
    bar = option_chain_df["QUOTE_READTIME"].iloc[0]
    vertical_spread = OptionStrategy.create_vertical_spread(
        symbol=config.ticker,
        option_type=STRATEGY_PARAMS["option_type"],
        long_strike=STRATEGY_PARAMS["long_delta"],
        short_strike=STRATEGY_PARAMS["short_delta"],
        expiration=STRATEGY_PARAMS["dte"],
        contracts=STRATEGY_PARAMS["contracts"],
        profit_target=STRATEGY_PARAMS["profit_target"],
        stop_loss=STRATEGY_PARAMS["stop_loss"],
        commission=STRATEGY_PARAMS["commission"],
        entry_time=bar,
        option_chain_df=option_chain_df,
        exit_scheme=exit_condition,
    )
    order = trading_manager.option_broker.create_order(vertical_spread)
    if trading_manager.add_order(order):
        logger.info(f"{bar}: Added order: {order}")
    else:
        logger.info(f"{bar}: Order not added {order}")

# Save the trading manager state
trading_manager.freeze(f"{cwd}/trading_manager60dte.pkl")
