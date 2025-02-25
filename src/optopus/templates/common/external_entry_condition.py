# MODULE TO DEFINE CUSTOM EXTERNAL ENTRY CONDITIONS
from optopus.trades.external_entry_conditions import (
    CompositePipelineCondition,
    ExternalEntryConditionChecker,
    LinearRegressionCheck,
    ParkinsonVolatilityDecreaseCheck,
    StdevVolatilityDecreaseCheck,
    YangZhangVolatilityDecreaseCheck,
    GarmanKlassVolatilityDecreaseCheck,
    RogersSatchellVolatilityDecreaseCheck,
    CloseToCloseVolatilityDecreaseCheck,
    VolatilityDecreaseCheck,
    IndicatorStateCheck,
)
import pandas as pd
import pandas_ta as pt


class MyAwesomeEntry(ExternalEntryConditionChecker):

    def __init__(self, **kwargs):
        self.linear_window = kwargs.get("linear_window", 14)
        self.lag1 = kwargs.get("lag1", 5)
        self.lag2 = kwargs.get("lag2", 10)
        self.gk_window = kwargs.get("gk_window", 30)
        self.yz_window = kwargs.get("yz_window", 30)
        self.c2c_window = kwargs.get("c2c_window", 30)
        self.zero_drift = kwargs.get("zero_drift", True)
        self.ohlc = kwargs.get("ohlc")
        
        self.pipeline = CompositePipelineCondition(
            LinearRegressionCheck(lag=self.linear_window)
            * IndicatorStateCheck(
                pt.ema, lag1=self.lag1, lag2=self.lag2
            )
            * GarmanKlassVolatilityDecreaseCheck(lag=self.gk_window)
            * YangZhangVolatilityDecreaseCheck(lag=self.yz_window)
            * CloseToCloseVolatilityDecreaseCheck(
                lag=self.c2c_window,
                zero_drift=self.zero_drift,
            ),
            self.ohlc,
        )

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.pipeline.should_enter(strategy, manager, time)

    def __repr__(self):
        return (
            f"MyAwesomeEntry(linear_window={self.linear_window}, "
            f"lag1={self.lag1}, lag2={self.lag2}, gk_window={self.gk_window}, "
            f"yz_window={self.yz_window}, c2c_window={self.c2c_window}, "
            f"zero_drift={self.zero_drift}, ohlc={self.ohlc})"
        )
