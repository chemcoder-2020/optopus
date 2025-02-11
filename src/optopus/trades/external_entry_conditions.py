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
        """
        Entry condition based on technical indicators and forecast models.

        Args:
            ohlc (str or pd.DataFrame): OHLC data file path or DataFrame
            atr_period (int, optional): Period for ATR calculation. Defaults to 14
            linear_regression_lag (int, optional): Lag period for linear regression check. Defaults to 14
            median_trend_short_lag (int, optional): Short period for median trend check. Defaults to 50
            median_trend_long_lag (int, optional): Long period for median trend check. Defaults to 200
        """
        self.data_processor = DataProcessor(
            kwargs.get("ohlc"), ticker=kwargs.get("ticker")
        )
        self.technical_indicators = TechnicalIndicators()
        self.forecast_models = ForecastModels()
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        current_price = strategy.underlying_last
        if not hasattr(self.data_processor, "ticker"):
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

        # Check technical indicators with bypass
        lin_reg_lag = self.kwargs.get("linear_regression_lag")
        if lin_reg_lag is not None:
            linear_trend = self.technical_indicators.check_linear_regression(
                historical_data, lag=lin_reg_lag
            )
            logger.debug(f"Linear regression trend check: {linear_trend}")
            if not linear_trend:
                logger.info("Entry rejected - failed linear regression trend check")
                return False
        else:
            logger.debug("Bypassing linear regression check")

        med_short = self.kwargs.get("median_trend_short_lag")
        med_long = self.kwargs.get("median_trend_long_lag")
        if med_short is not None or med_long is not None:
            median_trend = self.technical_indicators.check_median_trend(
                historical_data,
                short_lag=med_short if med_short is not None else 50,
                long_lag=med_long if med_long is not None else 200,
            )
            logger.debug(f"Median trend check: {median_trend}")
            if not median_trend:
                logger.info("Entry rejected - failed median trend check")
                return False
        else:
            logger.debug("Bypassing median trend check")

        # Check forecast models with bypass
        forecast_model = self.kwargs.get("forecast_model")
        if forecast_model is not None:
            if forecast_model == "arima":
                order = self.kwargs.get("order")
                seasonal_order = self.kwargs.get("seasonal_order")
                arima_trend = self.forecast_models.check_arima_trend(
                    monthly_data, current_price, 
                    order=order if order is not None else (0, 1, 1), 
                    seasonal_order=seasonal_order if seasonal_order is not None else (0, 1, 1)
                )
                logger.debug(f"ARIMA trend check: {arima_trend}")
                if not arima_trend:
                    logger.info("Entry rejected - failed ARIMA trend check")
                    return False
            elif forecast_model == "autoarima":
                autoarima_trend = self.forecast_models.check_autoarima_trend(
                    monthly_data, current_price
                )
                logger.debug(f"AutoARIMA trend check: {autoarima_trend}")
                if not autoarima_trend:
                    logger.info("Entry rejected - failed AutoARIMA trend check")
                    return False
            elif forecast_model == "autoces":
                autoces_trend = self.forecast_models.check_autoces_trend(
                    monthly_data, current_price
                )
                logger.debug(f"AutoCES trend check: {autoces_trend}")
                if not autoces_trend:
                    logger.info("Entry rejected - failed AutoCES trend check")
                    return False
            elif forecast_model == "nbeats":
                nbeats_trend = self.forecast_models.check_nbeats_trend(monthly_data)
                logger.debug(f"NBEATS trend check: {nbeats_trend}")
                if not nbeats_trend:
                    logger.info("Entry rejected - failed NBEATS trend check")
                    return False
            elif forecast_model in ["svm", "random_forest", "logistic", "gradient_boosting", "gaussian_process", "mlp", "knn"]:
                ml_trend = self.forecast_models.check_ML_trend(monthly_data, classifier=forecast_model)
                logger.debug(f"ML trend ({forecast_model}) check: {ml_trend}")
                if not ml_trend:
                    logger.info(f"Entry rejected - failed ML trend ({forecast_model}) check")
                    return False
            elif forecast_model == "oscillator":
                osc_lags = self.kwargs.get("oscillator_lags")
                oscillator_trend = self.forecast_models.check_seasonality_oscillator(
                    monthly_data, 
                    lags=osc_lags if osc_lags is not None else 3
                ) if osc_lags is not None else True
                logger.debug(f"Oscillator trend check: {oscillator_trend}")
                if not oscillator_trend:
                    logger.info("Entry rejected - failed oscillator trend check")
                    return False
            else:
                logger.info("Entry rejected - unknown forecast model")
                return False
        else:
            logger.debug("Bypassing forecast model checks")

        return True


class EntryOnForecastPlusKellyCriterion(ExternalEntryConditionChecker):
    def __init__(self, **kwargs):
        """
        Entry condition based on technical indicators and forecast models.

        Args:
            ohlc (str or pd.DataFrame): OHLC data file path or DataFrame
            atr_period (int, optional): Period for ATR calculation. Defaults to 14
            linear_regression_lag (int, optional): Lag period for linear regression check. Defaults to 14
            median_trend_short_lag (int, optional): Short period for median trend check. Defaults to 50
            median_trend_long_lag (int, optional): Long period for median trend check. Defaults to 200
            fractional_kelly (float, optional): Fractional Kelly factor. Defaults to 0.1
            n_lookback_kelly (int, optional): Lookback period for Kelly criterion. Defaults to 20
        """
        self.data_processor = DataProcessor(
            kwargs.get("ohlc"), ticker=kwargs.get("ticker")
        )
        self.technical_indicators = TechnicalIndicators()
        self.forecast_models = ForecastModels()
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        current_price = strategy.underlying_last
        
        # Handle Kelly criterion with bypass
        n_lookback = self.kwargs.get("n_lookback_kelly")
        fractional = self.kwargs.get("fractional_kelly")
        
        if n_lookback is not None and fractional is not None:
            if (
                len(manager.closed_trades) >= n_lookback
                and len(manager.closed_trades) % self.kwargs.get("kelly_update_interval", 1) == 0
            ):
                kc = manager.calculate_kelly_criterion(n_lookback, fractional)
                logger.debug(f"Calculated Kelly criterion: {kc}")
                
                if self.kwargs.get("min_position_size", None):
                    kc = max(kc, self.kwargs.get("min_position_size", 0))
                    logger.debug(f"Applied min position size constraint: {kc}")
                
                if self.kwargs.get("max_position_size", None):
                    kc = min(kc, self.kwargs.get("max_position_size", 0.1))
                    logger.debug(f"Applied max position size constraint: {kc}")

                if isinstance(kc, float) and 1 > kc > 0:
                    manager.update_config(position_size=kc)
                    logger.info(f"Updated position size to {kc} based on Kelly criterion")
        else:
            logger.debug("Bypassing Kelly criterion updates")
        
        if not hasattr(self.data_processor, "ticker"):
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

        # Check technical indicators with bypass
        lin_reg_lag = self.kwargs.get("linear_regression_lag")
        if lin_reg_lag is not None:
            linear_trend = self.technical_indicators.check_linear_regression(
                historical_data, lag=lin_reg_lag
            )
            logger.debug(f"Linear regression trend check: {linear_trend}")
            if not linear_trend:
                logger.info("Entry rejected - failed linear regression trend check")
                return False
        else:
            logger.debug("Bypassing linear regression check")

        # Check median trend with bypass
        med_short = self.kwargs.get("median_trend_short_lag")
        med_long = self.kwargs.get("median_trend_long_lag")
        if med_short is not None and med_long is not None:
            median_trend = self.technical_indicators.check_median_trend(
                historical_data,
                short_lag=med_short,
                long_lag=med_long,
            )
            logger.debug(f"Median trend check: {median_trend}")
            if not median_trend:
                logger.info("Entry rejected - failed median trend check")
                return False
        else:
            logger.debug("Bypassing median trend check")

        # Check forecast models with bypass
        forecast_model = self.kwargs.get("forecast_model")
        if forecast_model is not None:
            if forecast_model == "arima":
                arima_trend = self.forecast_models.check_arima_trend(
                    monthly_data, current_price
                )
                logger.debug(f"ARIMA trend check: {arima_trend}")
                if not arima_trend:
                    logger.info("Entry rejected - failed ARIMA trend check")
                    return False
            elif forecast_model == "autoarima":
                autoarima_trend = self.forecast_models.check_autoarima_trend(
                    monthly_data, current_price
                )
                logger.debug(f"AutoARIMA trend check: {autoarima_trend}")
                if not autoarima_trend:
                    logger.info("Entry rejected - failed AutoARIMA trend check")
                    return False
            elif forecast_model == "autoces":
                autoces_trend = self.forecast_models.check_autoces_trend(
                    monthly_data, current_price
                )
                logger.debug(f"AutoCES trend check: {autoces_trend}")
                if not autoces_trend:
                    logger.info("Entry rejected - failed AutoCES trend check")
                    return False
            elif forecast_model == "nbeats":
                nbeats_trend = self.forecast_models.check_nbeats_trend(monthly_data)
                logger.debug(f"NBEATS trend check: {nbeats_trend}")
                if not nbeats_trend:
                    logger.info("Entry rejected - failed NBEATS trend check")
                    return False
            else:
                logger.info("Entry rejected - unknown forecast model")
                return False
        else:
            logger.debug("Bypassing forecast model checks")

        return True
