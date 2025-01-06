from optopus.trades.entry_conditions import (
    EntryConditionChecker,
    CompositeEntryCondition,
    CapitalRequirementCondition,
    PositionLimitCondition,
    RORThresholdCondition,
    ConflictCondition,
)

import pandas as pd
import numpy as np
from optopus.utils.heapmedian import ContinuousMedian
import os
from loguru import logger
from optopus.brokers.schwab.schwab_data import SchwabData


class MedianCalculator(EntryConditionChecker):
    def __init__(self, window_size=7, fluctuation=0.1):
        self.median_calculator = ContinuousMedian()
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.premiums = []

    def add_premium(self, mark):
        self.median_calculator.add(mark)
        self.premiums.append(mark)
        if len(self.premiums) > self.window_size:
            self.median_calculator.remove(self.premiums.pop(0))

    def get_median(self):
        return self.median_calculator.get_median()

    def should_enter(self, strategy, manager, time) -> bool:
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2

        self.add_premium(mark)
        median_mark = self.get_median()

        return np.isclose(mark, median_mark, rtol=self.fluctuation) and np.isclose(
            bid, mark, rtol=self.fluctuation
        )

class EntryCondition(EntryConditionChecker):
    def __init__(self, **kwargs):
        self.composite = CompositeEntryCondition(
            [
                CapitalRequirementCondition(),
                PositionLimitCondition(),
                RORThresholdCondition(),
                ConflictCondition(
                    check_closed_trades=kwargs.get("check_closed_trades", True)
                ),
                MedianCalculator(
                    window_size=kwargs.get("window_size", 7),
                    fluctuation=kwargs.get("fluctuation", 0.1),
                ),
            ]
        )
        self.recent_marks = []
        self.ohlc = kwargs.get("ohlc")
        if type(self.ohlc) is str:
            self.ohlc = pd.read_csv(self.ohlc, parse_dates=["date"]).set_index("date")
        self.daily_data = (
            self.ohlc.resample("D")
            .apply(
                {
                    "close": "last",
                    "open": "first",
                    "low": "min",
                    "high": "max",
                    "volume": "sum",
                }
            )
            .dropna(subset="close")
        )
        self.daily_data.reset_index(inplace=True)
        self.daily_data.rename(columns={"date": "day"}, inplace=True)

        self.daily_close = self.daily_data.set_index("day").close
        self.daily_high = self.daily_data.set_index("day").high
        self.daily_low = self.daily_data.set_index("day").low
        self.daily_volume = self.daily_data.set_index("day").volume

        self.daily_data["previous_day"] = self.daily_data["day"].shift(1)

        self.ohlc = self.ohlc.reset_index()
        self.ohlc["day"] = pd.DatetimeIndex(self.ohlc.date.dt.date)
        self.ohlc = self.ohlc.merge(
            self.daily_data[["day", "previous_day"]], on="day", how="left"
        )
        self.ohlc.rename(columns={"previous_day": "previous_date"}, inplace=True)
        self.ohlc["next_candle"] = self.ohlc["date"].shift(-1)

        def median(close, length=50, **kwargs):
            return close.rolling(length).median()

        self.ohlc["Median100"] = self.aggregate(
            median, length=100, previous_lookback_days=100
        )
        self.ohlc["Median200"] = self.aggregate(
            median, length=200, previous_lookback_days=200
        )

    def aggregate(self, indicator_fn, **kwargs):
        length = kwargs.get("length")
        previous_lookback_days = kwargs.get("previous_lookback_days", 200)

        def aggregate_fn(x):
            d = x["previous_date"]
            close = pd.Series(
                self.daily_close.loc[:d][-previous_lookback_days:].to_list() + [x.close]
            )
            high = pd.Series(
                self.daily_high.loc[:d][-previous_lookback_days:].to_list() + [x.high]
            )
            low = pd.Series(
                self.daily_low.loc[:d][-previous_lookback_days:].to_list() + [x.low]
            )
            if len(close) < previous_lookback_days:
                return np.nan
            else:
                e = indicator_fn(high=high, low=low, close=close, length=length)
                return e.iloc[-1]

        return self.ohlc.apply(aggregate_fn, axis=1, result_type="expand")

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        df = self.ohlc.set_index("next_candle")

        basic_condition = self.composite.should_enter(strategy, manager, time)

        try:
            return (
                df.loc[time, "Median100"] > df.loc[time, "Median200"]
                and basic_condition
            )
        except KeyError:
            return False


class BotEntryCondition(EntryConditionChecker):
    def __init__(self, **kwargs):
        self.composite = CompositeEntryCondition(
            [
                CapitalRequirementCondition(),
                PositionLimitCondition(),
                RORThresholdCondition(),
                ConflictCondition(check_closed_trades=True),
            ]
        )
        self.ticker = kwargs.get("ticker")
        self.basename = kwargs.get("basename")

    def should_enter(self, strategy, manager, time):
        ticker = self.ticker
        basename = self.basename

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

        SMA100 = indicator_prices.rolling(100).mean().iloc[-1]
        SMA200 = indicator_prices.rolling(200).mean().iloc[-1]
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
