from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING
from loguru import logger
from ..utils.ohlc_data_processor import DataProcessor
from ..decisions.technical_indicators import TechnicalIndicators
from ..decisions.forecast_models import ForecastModels

if TYPE_CHECKING:
    from .option_manager import OptionBacktester


class ExternalEntryConditionChecker(ABC):
    """
    Abstract base class for external entry condition checkers.
    These conditions can be used alongside standard EntryConditionCheckers
    to implement additional custom logic for trade entry decisions.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Check if the external entry conditions are met for the option strategy.
    """

    @abstractmethod
    def should_enter(
        self,
        time: Union[datetime, str, pd.Timestamp],
        strategy=None,
        manager: "OptionBacktester" = None,
    ) -> bool:
        """
        Check if the external entry conditions are met.

        Args:
            time: The current time of evaluation (required)
            strategy: The option strategy being evaluated (optional)
            manager: The option backtester/manager instance (optional)

        Returns:
            bool: True if external conditions are met, False otherwise
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the condition checker"""
        return f"{self.__class__.__name__}()"


class EntryOnForecast(ExternalEntryConditionChecker):
    def __init__(self, **kwargs):
        self.data_processor = DataProcessor(kwargs.get("ohlc"))
        self.technical_indicators = TechnicalIndicators()
        self.forecast_models = ForecastModels()
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        current_price = strategy.underlying_last
        self.data_processor.ticker = strategy.symbol

        historical_data, monthly_data = self.data_processor.prepare_historical_data(
            time, current_price
        )

        # Calculate and store ATR
        manager.atr = self.technical_indicators.calculate_atr(
            historical_data["high"],
            historical_data["low"],
            historical_data["close"],
            period=self.kwargs.get("atr_period", 14),
        )

        # Check technical indicators
        linear_trend = self.technical_indicators.check_linear_regression(
            historical_data, lag=self.kwargs.get("linear_regression_lag", 14)
        )
        if not linear_trend:
            return False

        median_trend = self.technical_indicators.check_median_trend(
            historical_data,
            short_lag=self.kwargs.get("median_trend_short_lag", 50),
            long_lag=self.kwargs.get("median_trend_long_lag", 200),
        )
        if not median_trend:
            return False

        # Check forecast models
        arima_trend = self.forecast_models.check_arima_trend(
            monthly_data, current_price
        )
        if not arima_trend:
            return False

        return True
