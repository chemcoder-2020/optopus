from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING
from loguru import logger
from ...utils.ohlc_data_processor import DataProcessor
from ...decisions.technical_indicators import TechnicalIndicators
from ...decisions.forecast_models import ForecastModels

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


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
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
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

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.left.should_enter(
            time=time, strategy=strategy, manager=manager
        ) & self.right.should_enter(time=time, strategy=strategy, manager=manager)


class OrComponent(BaseComponent):
    """OR logical operator component"""

    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.left.should_enter(
            time=time, strategy=strategy, manager=manager
        ) | self.right.should_enter(time=time, strategy=strategy, manager=manager)


class NotComponent(BaseComponent):
    """NOT logical operator component"""

    def __init__(self, component: BaseComponent):
        super().__init__()
        self.component = component

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return not self.component.should_enter(
            time=time, strategy=strategy, manager=manager
        )


@BaseComponent.register("rsi")
class IndicatorCheck(BaseComponent):
    """Component for technical indicator checks"""

    def __init__(self, name: str, **params):
        self.name = name.lower()
        self.params = params
        self._validate_indicator()

    def _validate_indicator(self):
        valid_indicators = {
            "atr": (
                TechnicalIndicators.calculate_atr,
                ["high", "low", "close", "period"],
            ),
            "linear_regression": (
                TechnicalIndicators.check_linear_regression,
                ["historical_data", "lag"],
            ),
            "median_trend": (
                TechnicalIndicators.check_median_trend,
                ["historical_data", "short_lag", "long_lag"],
            ),
            "rsi": (
                TechnicalIndicators.check_rsi,
                ["historical_data", "period", "oversold"],
            ),
        }
        if self.name not in valid_indicators:
            raise ValueError(f"Invalid indicator: {self.name}")
        self.func, self.expected_args = valid_indicators[self.name]

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        bound_args = self._bind_arguments()
        return self.func(TechnicalIndicators, **bound_args)

    def _bind_arguments(self):
        bound_args = {}
        for arg in self.expected_args:
            if arg == "historical_data":
                # Get historical_data from params if not in manager context
                bound_args[arg] = self.params.get("historical_data")
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
            "arima": (
                ForecastModels.check_arima_trend,
                ["monthly_data", "current_price", "order", "seasonal_order"],
            ),
            "autoarima": (
                ForecastModels.check_autoarima_trend,
                ["monthly_data", "current_price"],
            ),
            "ml_model": (ForecastModels.check_ML_trend, ["monthly_data", "classifier"]),
        }
        if self.name not in valid_models:
            raise ValueError(f"Invalid model: {self.name}")
        self.func, self.expected_args = valid_models[self.name]

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        bound_args = self._bind_arguments(manager)
        return self.func(ForecastModels, **bound_args)

    def _bind_arguments(self, manager):
        bound_args = {}
        for arg in self.expected_args:
            if arg == "monthly_data":
                bound_args[arg] = manager.context.get("monthly_data")
            elif arg == "current_price":
                bound_args[arg] = manager.context.get("current_price")
            else:
                bound_args[arg] = self.params.get(arg)
        return bound_args

    def __repr__(self):
        return f"ModelCheck(name={self.name}, params={self.params})"


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

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        logger.debug(f"Evaluating pipeline at {time}: {self.pipeline}")

        # Prepare market data
        current_price = strategy.underlying_last if strategy else None
        try:
            hist_data, monthly_data = self.data_processor.prepare_historical_data(time)
        except Exception as e:
            logger.error(f"Error preparing historical data: {e}")
            return False

        # Initialize context if needed
        if not hasattr(manager, "context"):
            logger.debug("Creating new context in manager")
            manager.context = {}

        if "indicators" not in manager.context:
            manager.context["indicators"] = {}

        # Store data in context for components
        manager.context.update(
            {
                "historical_data": hist_data,
                "monthly_data": monthly_data,
                "current_price": current_price,
                "bar": time,
            }
        )
        logger.debug(
            f"Context at {time} updated with historical data ({len(hist_data)} rows), monthly data ({len(monthly_data)} rows), current price {current_price}"
        )

        # Evaluate the pipeline
        try:
            result = self.pipeline.should_enter(
                time=time, strategy=strategy, manager=manager
            )
        except Exception as e:
            logger.error(f"Error evaluating pipeline: {e}")
            result = False
        logger.debug(f"Pipeline evaluation result: {result}")
        return result
