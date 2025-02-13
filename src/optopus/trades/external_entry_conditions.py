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


class BaseComponent:
    """Base class for all pipeline components with operator overloading"""
    _registry = {}  # Class-level component registry
    
    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs):
        """Factory method for creating components"""
        return cls._registry[name.lower()](**kwargs)

    def __mul__(self, other):
        return AndComponent(self, other)
    
    def __or__(self, other):
        return OrComponent(self, other)
    
    def __invert__(self):
        return NotComponent(self)

class AndComponent(BaseComponent):
    """AND logical operator component"""
    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        return (
            self.left.should_enter(time, strategy, manager) and 
            self.right.should_enter(time, strategy, manager)
        )

class OrComponent(BaseComponent):
    """OR logical operator component"""
    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        return (
            self.left.should_enter(time, strategy, manager) or 
            self.right.should_enter(time, strategy, manager)
        )

class NotComponent(BaseComponent):
    """NOT logical operator component"""
    def __init__(self, component: BaseComponent):
        super().__init__()
        self.component = component

    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        return not self.component.should_enter(time, strategy, manager)

@BaseComponent.register("rsi")
class IndicatorCheck(BaseComponent):
    """Component for technical indicator checks"""
    def __init__(self, name: str, **params):
        self.name = name.lower()
        self.params = params
        self._validate_indicator()
        
    def _validate_indicator(self):
        valid_indicators = {
            'atr': (TechnicalIndicators.calculate_atr, ['high', 'low', 'close', 'period']),
            'linear_regression': (TechnicalIndicators.check_linear_regression, ['historical_data', 'lag']),
            'median_trend': (TechnicalIndicators.check_median_trend, ['historical_data', 'short_lag', 'long_lag']),
            'rsi': (TechnicalIndicators.check_rsi, ['historical_data', 'period', 'oversold'])
        }
        if self.name not in valid_indicators:
            raise ValueError(f"Invalid indicator: {self.name}")
        self.func, self.expected_args = valid_indicators[self.name]
        
    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        hist_data = manager.context['historical_data']
        bound_args = self._bind_arguments()
        return self.func(TechnicalIndicators, **bound_args)
    
    def _bind_arguments(self):
        bound_args = {}
        for arg in self.expected_args:
            if arg == 'historical_data':
                # Get historical_data from params if not in manager context
                bound_args[arg] = self.params.get('historical_data')
            else:
                bound_args[arg] = self.params.get(arg)
        return bound_args
        
    def __repr__(self):
        return f"IndicatorCheck(name={self.name}, params={self.params})"

class ModelCheck(BaseComponent):
    """Component for forecast model checks"""
    def __init__(self, name: str, **params):
        self.name = name.lower()
        self.params = params
        self._validate_model()
        
    def _validate_model(self):
        valid_models = {
            'arima': (ForecastModels.check_arima_trend, ['monthly_data', 'current_price', 'order', 'seasonal_order']),
            'autoarima': (ForecastModels.check_autoarima_trend, ['monthly_data', 'current_price']),
            'ml_model': (ForecastModels.check_ML_trend, ['monthly_data', 'classifier'])
        }
        if self.name not in valid_models:
            raise ValueError(f"Invalid model: {self.name}")
        self.func, self.expected_args = valid_models[self.name]
        
    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        bound_args = self._bind_arguments(manager)
        return self.func(ForecastModels, **bound_args)
    
    def _bind_arguments(self, manager):
        bound_args = {}
        for arg in self.expected_args:
            if arg == 'monthly_data':
                bound_args[arg] = manager.context.get('monthly_data')
            elif arg == 'current_price':
                bound_args[arg] = manager.context.get('current_price')
            else:
                bound_args[arg] = self.params.get(arg)
        return bound_args
        
    def __repr__(self):
        return f"ModelCheck(name={self.name}, params={self.params})"

class RSIComponent(BaseComponent):
    """RSI component with operator support"""
    def __init__(self, period=14, oversold=30):
        self.period = period
        self.oversold = oversold
        
    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        hist_data = manager.context['historical_data']
        return TechnicalIndicators.check_rsi(
            historical_data=hist_data,
            period=self.period,
            oversold=self.oversold
        )
        
    def __repr__(self):
        return f"RSIComponent(period={self.period}, oversold={self.oversold})"

class CompositePipelineCondition(ExternalEntryConditionChecker):
    """Decision pipeline combining multiple indicators/models using logical operators"""
    def __init__(self, pipeline: BaseComponent, ohlc_data: str, ticker=None):
        """
        Args:
            pipeline: Configured pipeline using component operators
            ohlc_data: Path to OHLC data file
        """
        self.pipeline = pipeline
        self.data_processor = DataProcessor(ohlc_data, ticker=ticker)
        
    def should_enter(self, time: pd.Timestamp, strategy=None, manager=None) -> bool:
        # Prepare market data
        current_price = strategy.underlying_last if strategy else None
        hist_data, monthly_data = self.data_processor.prepare_historical_data(time, current_price)
        
        # Initialize context if needed
        if not hasattr(manager, 'context'):
            manager.context = {}
            
        # Store data in context for components
        manager.context.update({
            'historical_data': hist_data,
            'monthly_data': monthly_data,
            'current_price': current_price
        })
        
        # Evaluate the pipeline
        return self.pipeline.should_enter(time, strategy, manager)

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
        # Validate linear regression params
        lin_reg_lag = self.kwargs.get("linear_regression_lag")
        if lin_reg_lag in ("", None):  # Handle empty string as None
            lin_reg_lag = None
        elif lin_reg_lag is not None and not isinstance(lin_reg_lag, int):
            logger.warning(f"Invalid linear_regression_lag type {type(lin_reg_lag)}, expected int. Bypassing check")
            lin_reg_lag = None
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

        # Validate median trend params
        med_short = self.kwargs.get("median_trend_short_lag")
        med_long = self.kwargs.get("median_trend_long_lag")
        
        if med_short in ("", None):  # Handle empty string as None
            med_short = None
        elif med_short is not None and not isinstance(med_short, int):
            logger.warning(f"Invalid median_trend_short_lag type {type(med_short)}, expected int. Bypassing check")
            med_short = None
        if med_long in ("", None):  # Handle empty string as None
            med_long = None
        elif med_long is not None and not isinstance(med_long, int):
            logger.warning(f"Invalid median_trend_long_lag type {type(med_long)}, expected int. Bypassing check")
            med_long = None
            
        if med_short is not None and med_long is not None:
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
        # Validate forecast model params
        forecast_model = self.kwargs.get("forecast_model")
        if forecast_model in ("", None):  # Explicitly handle empty string
            forecast_model = None
        elif forecast_model is not None and not isinstance(forecast_model, str):
            logger.warning(f"Invalid forecast_model type {type(forecast_model)}, expected str. Bypassing check")
            forecast_model = None
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
        # Validate Kelly params
        n_lookback = self.kwargs.get("n_lookback_kelly")
        fractional = self.kwargs.get("fractional_kelly")
        
        if n_lookback in ("", None):  # Handle empty string as None
            n_lookback = None
        elif n_lookback is not None and not isinstance(n_lookback, int):
            logger.warning(f"Invalid n_lookback_kelly type {type(n_lookback)}, expected int. Bypassing Kelly updates")
            n_lookback = None
        if fractional in ("", None):  # Handle empty string as None
            fractional = None
        elif fractional is not None and not isinstance(fractional, (float, int)):
            logger.warning(f"Invalid fractional_kelly type {type(fractional)}, expected float/int. Bypassing Kelly updates")
            fractional = None
        
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

        # Check technical indicators with bypass and validation
        lin_reg_lag = self.kwargs.get("linear_regression_lag")
        if lin_reg_lag in ("", None):  # Handle empty string as None
            lin_reg_lag = None
        elif lin_reg_lag is not None and not isinstance(lin_reg_lag, int):
            logger.warning(f"Invalid linear_regression_lag type {type(lin_reg_lag)}, expected int. Bypassing check")
            lin_reg_lag = None
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

        # Check median trend with bypass and validation
        med_short = self.kwargs.get("median_trend_short_lag")
        med_long = self.kwargs.get("median_trend_long_lag")
        
        if med_short in ("", None):  # Handle empty string as None
            med_short = None
        elif med_short is not None and not isinstance(med_short, int):
            logger.warning(f"Invalid median_trend_short_lag type {type(med_short)}, expected int. Bypassing check")
            med_short = None
        if med_long in ("", None):  # Handle empty string as None
            med_long = None
        elif med_long is not None and not isinstance(med_long, int):
            logger.warning(f"Invalid median_trend_long_lag type {type(med_long)}, expected int. Bypassing check")
            med_long = None
            
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
        if forecast_model in ("", None):  # Handle empty string explicitly
            forecast_model = None
        elif forecast_model is not None and not isinstance(forecast_model, str):
            logger.warning(f"Invalid forecast_model type {type(forecast_model)}, expected str. Bypassing check")
            forecast_model = None
        if forecast_model is not None:
            if forecast_model == "arima":
                order = self.kwargs.get("order")
                seasonal_order = self.kwargs.get("seasonal_order")
                arima_trend = self.forecast_models.check_arima_trend(
                    monthly_data, current_price, 
                    order=order if order is not None else (0, 1, 1), 
                    seasonal_order=seasonal_order if seasonal_order is not None else (0, 1, 1)
                )
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
