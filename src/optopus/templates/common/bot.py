import os
import dill
from optopus.trades.trade_manager import TradingManager
import pandas as pd
import dotenv
from loguru import logger

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
basename = os.path.split(dname)[0]

from config import (
    STRATEGY_PARAMS,
    BACKTESTER_CONFIG,
)
from entry_condition import BotEntryCondition


# Configure loguru to write logs to a file
logger.add(
    f"{cwd}/60dteBot.log", rotation="10 MB", retention="60 days", compression="zip"
)
logger.enable("optopus")

pd.options.display.max_columns = 50

dotenv.load_dotenv(os.path.join(basename, ".env"))




# Initialize the trading manager
if os.path.exists(f"{cwd}/trading_manager60dte.pkl"):
    trading_manager = dill.load(open(f"{cwd}/trading_manager60dte.pkl", "rb"))
else:
    config = BACKTESTER_CONFIG
    ticker = STRATEGY_PARAMS["symbol"]
    config.ticker = ticker
    config.broker = "Schwab"
    config.client_id = os.getenv("SCHWAB_CLIENT_ID")
    config.client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    config.redirect_uri = os.getenv("SCHWAB_REDIRECT_URI")
    config.token_file = os.path.join(basename, "token.json")
    config.entry_condition = BotEntryCondition(ticker=ticker, basename=basename)
    trading_manager = TradingManager(config)

trading_manager.auth_refresh()

trading_manager.next(STRATEGY_PARAMS)

trading_manager.freeze(f"{cwd}/trading_manager60dte.pkl")
