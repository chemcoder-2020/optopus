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
        self.pipeline = CompositePipelineCondition(
            LinearRegressionCheck(lag=kwargs.get("linear_window", 14))
            * IndicatorStateCheck(
                pt.ema, lag1=kwargs.get("lag1", 5), lag2=kwargs.get("lag2", 10)
            )
            * GarmanKlassVolatilityDecreaseCheck(lag=kwargs.get("gk_window", 30))
            * YangZhangVolatilityDecreaseCheck(lag=kwargs.get("yz_window", 30))
            * CloseToCloseVolatilityDecreaseCheck(
                lag=kwargs.get("c2c_window", 30),
                zero_drift=kwargs.get("zero_drift", True),
            ),
            kwargs.get("ohlc"),
        )

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.pipeline.should_enter(strategy, manager, time)

    def __repr__(self):
        return f"MyAwesomeEntry()"
