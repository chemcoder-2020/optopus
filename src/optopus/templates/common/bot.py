import os
import dill
from optopus.trades.trade_manager import TradingManager
from optopus.trades.option_manager import Config
from optopus.trades.entry_conditions import (
    EntryConditionChecker,
    CompositeEntryCondition,
    CapitalRequirementCondition,
    PositionLimitCondition,
    RORThresholdCondition,
    ConflictCondition
)
from optopus.brokers.schwab.schwab_data import SchwabData
from optopus.trades.exit_conditions import DefaultExitCondition
import pandas as pd
import numpy as np
import pandas_ta as pt
import dotenv
from loguru import logger
from config import (
    DATA_FOLDER,
    START_DATE,
    END_DATE,
    TRADING_START_TIME,
    TRADING_END_TIME,
    DEBUG,
    STRATEGY_PARAMS,
    BACKTESTER_CONFIG
)

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

dotenv.load_dotenv(os.path.join(basename, ".env"))

logger.enable("optopus")

ticker = STRATEGY_PARAMS["symbol"]


class EntryCondition(EntryConditionChecker):
    def __init__(self):
        self.composite = CompositeEntryCondition(
            [
                CapitalRequirementCondition(),
                PositionLimitCondition(),
                RORThresholdCondition(),
                ConflictCondition(check_closed_trades=True),
            ]
        )

    def should_enter(self, strategy, manager, time):

        self.schwab_data = SchwabData(
            client_id=os.getenv("SCHWAB_CLIENT_ID"),
            client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
            redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
            token_file=os.path.join(basename, "token.json"),
        )
        self.schwab_data.refresh_token()
        time = pd.Timestamp(time)
        equity_price = self.schwab_data.get_price_history(
            ticker, "year", 2, frequency_type="daily", frequency=1
        )
        current_quote = self.schwab_data.get_quote(ticker)
        indicator_prices = pd.concat(
            [
                equity_price,
                pd.concat(
                    [current_quote["MARK"], current_quote["QUOTE_TIME"].dt.date], axis=1
                ).rename(columns={"MARK": "close", "QUOTE_TIME": "datetime"}),
            ],
            axis=0,
            ignore_index=True,
        ).set_index("datetime")["close"]

        SMA100 = pt.sma(indicator_prices, length=100).iloc[-1]
        SMA200 = pt.sma(indicator_prices, length=200).iloc[-1]
        Median100 = indicator_prices.rolling(100).median().iloc[-1]
        Median200 = indicator_prices.rolling(200).median().iloc[-1]
        logger.info(
            f"SMA100: {SMA100}; SMA200: {SMA200}; Median100: {Median100}; Median200: {Median200}"
        )

        logger.info(strategy)
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (bid + ask) / 2
        logger.info(f"Mark price at {mark}")
        if hasattr(manager, "premiums"):
            manager.premiums.append(mark)
            if len(manager.premiums) > 25:
                manager.premiums.pop(0)
        else:
            manager.premiums = []

        median_mark = np.median(manager.premiums)
        price_condition = np.isclose(mark, median_mark, rtol=0.1) and np.isclose(
            bid, mark, rtol=0.1
        )
        basic_condition = self.composite.should_enter(strategy, manager, time)
        logger.info(
            f"Price Entry Condition: {price_condition}; Median mark: {median_mark}; Mark: {mark}; Bid: {bid}, Ask: {ask}"
        )
        logger.info(f"Basic Entry Condition: {basic_condition}")
        return Median100 > Median200 and price_condition and basic_condition


# Initialize the trading manager
if os.path.exists(f"{cwd}/trading_manager60dte.pkl"):
    trading_manager = dill.load(open(f"{cwd}/trading_manager60dte.pkl", "rb"))
else:
    config = BACKTESTER_CONFIG
    config.ticker = ticker
    config.broker = "Schwab"
    config.client_id = os.getenv("SCHWAB_CLIENT_ID")
    config.client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    config.redirect_uri = os.getenv("SCHWAB_REDIRECT_URI")
    config.token_file = os.path.join(basename, "token.json")
    config.entry_condition = EntryCondition()
    config.trade_type = "Vertical Spread"
    trading_manager = TradingManager(config)

trading_manager.auth_refresh()

trading_manager.next(STRATEGY_PARAMS)

trading_manager.freeze(f"{cwd}/trading_manager60dte.pkl")
