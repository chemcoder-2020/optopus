from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, List, Type
from loguru import logger
import numpy as np
from ...utils.filters import HampelFilterNumpy, Filter
from ...utils.heapmedian import ContinuousMedian
import importlib


class ExitConditionChecker(ABC):
    """
    Abstract base class for exit condition checkers.

    Methods:
        should_exit(strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame, manager=None) -> bool:
            Check if the exit conditions are met for the option strategy.
    """

    def __repr__(self):
        """
        Return a string representation of the exit condition checker.

        Returns:
            str: String representation of the exit condition checker.
        """
        return f"{self.__class__.__name__}()"

    def update(self, **kwargs):
        """
        Update the attributes of the exit condition checker.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        """
        Check if the exit conditions are met for the option strategy.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
            manager (Optional[OptionBacktester]): The backtester instance managing the strategy. Defaults to None.

        Returns:
            bool: True if the exit conditions are met, False otherwise.
        """
        pass


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

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        # Docstring update not strictly needed here as it inherits, but adding manager to call
        return self.left.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        ) and self.right.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        )


class OrComponent(BaseComponent):
    """OR logical operator component"""

    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        # Docstring update not strictly needed here as it inherits, but adding manager to call
        return self.left.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        ) or self.right.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        )


class NotComponent(BaseComponent):
    """NOT logical operator component"""

    def __init__(self, component: BaseComponent):
        super().__init__()
        self.component = component

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        # Docstring update not strictly needed here as it inherits, but adding manager to call
        return not self.component.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        )


class MedianCalculator:
    """
    Class for calculating the median of a rolling window of premiums.

    Attributes:
        window_size (int): The size of the rolling window.
        premiums (list): List of premiums.
        median_calculator (ContinuousMedian): Continuous median calculator.
    """

    def __init__(self, window_size=5, method="ContinuousMedian", **kwargs):
        """
        Initialize the MedianCalculator.

        Args:
            window_size (int): The size of the rolling window.
            method (str): The method for calculating the median ('ContinuousMedian' or 'HampelFilter').
        Kwargs:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.window_size = window_size
        self.method = method
        self.kwargs = kwargs
        if method == "ContinuousMedian":
            self.median_calculator = ContinuousMedian()
        else:
            self.median_calculator = HampelFilterNumpy(
                window_size=window_size,
                n_sigma=self.kwargs.get("n_sigma", 3),
                k=self.kwargs.get("k", 1.4826),
                max_iterations=self.kwargs.get("max_iterations", 5),
                replace_with_na=self.kwargs.get("replace_with_na", False),
            )

        self.premiums = []

    def add_premium(self, mark):
        """
        Add a new premium to the rolling window.

        Args:
            mark (float): The new premium.
        """
        if self.method == "ContinuousMedian":
            self.median_calculator.add(mark)
        self.premiums.append(mark)

        if self.method == "ContinuousMedian":
            if len(self.premiums) > self.window_size:
                self.median_calculator.remove(self.premiums.pop(0))
        else:
            if len(self.premiums) > self.window_size + 1:
                self.premiums.pop(0)

    def get_median(self):
        """
        Get the current median of the rolling window.

        Returns:
            float: The current median.
        """
        if self.method == "ContinuousMedian":
            return self.median_calculator.get_median()
        else:
            if len(self.premiums) < self.window_size + 1:
                return 0
            else:
                return self.median_calculator.fit_transform(
                    np.array(self.premiums)
                ).flatten()[-1]

    def update(self, **kwargs):
        """
        Update the attributes of the MedianCalculator.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        if "window_size" in kwargs:
            new_window_size = kwargs["window_size"]
            # Adjust the window size while maintaining existing data
            if new_window_size < self.window_size:
                # Remove oldest entries if new window is smaller
                if self.method == "ContinuousMedian":
                    self.premiums = self.premiums[-new_window_size:]
                else:
                    self.premiums = self.premiums[-new_window_size - 1 :]
            self.window_size = new_window_size

    def get_median_return_percentage(self, strategy):
        """
        Calculate the median return percentage for the given strategy.

        Args:
            strategy (OptionStrategy): The option strategy to calculate the return percentage for.

        Returns:
            float: The median return percentage or direct return if no method specified
        """
        current_return = strategy.return_percentage()

        if (
            not hasattr(self, "method") or not self.method
        ):  # Handle empty/null method case
            strategy.filter_return_percentage = current_return
            strategy.filter_pl = strategy.total_pl()
            strategy.premium_log = self.premiums.copy()
            return current_return

        if self.premiums == []:  # take care of first update when adding spread
            self.add_premium(0)
        self.add_premium(current_return)
        median_return = self.get_median()
        strategy.premium_log = self.premiums.copy()
        strategy.filter_return_percentage = median_return
        strategy.filter_pl = (
            strategy.entry_net_premium * strategy.contracts
        ) * strategy.filter_return_percentage
        return median_return


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass


class PremiumListInit(Preprocessor):
    def preprocess(self, strategy):
        if not hasattr(strategy, "premium_log"):
            strategy.premium_log = []
            strategy.premium_bid = []
            strategy.premium_ask = []
            strategy.premium_qualified = []
            strategy.time_log = []
            logger.debug("Premium log initialized")
            return True


class PremiumFilter(Preprocessor):
    def __init__(
        self,
        filter_method: Union[Filter, Type[Filter], str] = HampelFilterNumpy,
        **kwargs,
    ):
        self._method = filter_method
        if isinstance(filter_method, str):
            filter_module = importlib.import_module("optopus.utils.filters")
            filter_method = getattr(filter_module, filter_method)
        elif isinstance(filter_method, type) and issubclass(filter_method, Filter):
            filter_method = filter_method
        else:
            raise ValueError(
                "filter_method must be a Filter class or a string name of a Filter class"
            )

        self.filter_method = filter_method
        self.kwargs = kwargs

        self.premium_filter = self.filter_method(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def preprocess(self, strategy):
        """
        Check if current premium's return percentage is an outlier. Updates strategy's filtered metrics with the cleaned values.

        Args:
            strategy (OptionStrategy): The option strategy to check
        """
        current_return = strategy.return_percentage()
        strategy.premium_log.append(current_return)
        strategy.premium_bid.append(strategy.current_bid)
        strategy.premium_ask.append(strategy.current_ask)
        strategy.time_log.append(strategy.current_time)

        filtered_returns = self.premium_filter.fit_transform(strategy.premium_log)

        # Store results in strategy
        strategy.filter_return_percentage = filtered_returns[-1]
        strategy.filter_pl = (
            strategy.entry_net_premium
            * strategy.contracts
            * strategy.filter_return_percentage
        )
        strategy.premium_qualified.append(True)

        return True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.filter_method.__name__}: {self.kwargs})"
        )


class CompositePipelineCondition(ExitConditionChecker):
    """Decision pipeline combining multiple indicators/models using logical operators"""

    def __init__(
        self,
        pipeline: BaseComponent,
        preprocessors: Union[Preprocessor, List[Preprocessor]] = None,
        **kwargs,
    ):
        """
        Args:
            pipeline: Configured pipeline using component operators
        """
        self.pipeline = pipeline
        self.preprocessors = preprocessors
        self.kwargs = kwargs

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:

        # Pre-processing
        if self.preprocessors:
            if isinstance(self.preprocessors, list):
                for preprocessor in self.preprocessors:
                    preprocessor.preprocess(strategy)
                    logger.debug(
                        f"{preprocessor.__class__.__name__} preprocessed strategy: {strategy} at {current_time}"
                    )
            else:
                self.preprocessors.preprocess(strategy)
                logger.debug(
                    f"{self.preprocessors.__class__.__name__} preprocessed strategy: {strategy} at {current_time}"
                )

        logger.debug(f"Preprocessed strategy at {current_time}")

        # Evaluate the pipeline
        result = self.pipeline.should_exit(
            strategy=strategy,
            current_time=current_time,
            option_chain_df=option_chain_df,
            manager=manager,
        )
        if result:
            if np.isnan(strategy.filter_pl):
                strategy.filter_pl = strategy.total_pl()

            if np.isnan(strategy.filter_return_percentage):
                strategy.filter_return_percentage = strategy.return_percentage()
        logger.debug(f"Pipeline evaluation result: {result}")
        return result


class CompositeExitCondition(ExitConditionChecker):
    """
    Composite exit condition that combines multiple exit conditions.

    Attributes:
        conditions (List[ExitConditionChecker]): List of exit conditions to combine.
        logical_operations (List[str]): List of logical operations to combine the conditions ('AND' or 'OR').
    """

    def __init__(
        self,
        conditions: List[ExitConditionChecker],
        logical_operations: List[str] = ["AND"],
    ):
        """
        Initialize the CompositeExitCondition.

        Args:
            conditions (List[ExitConditionChecker]): List of exit conditions to combine.
            logical_operations (List[str]): List of logical operations to combine the conditions ('AND' or 'OR').
        """
        self.conditions = conditions
        self.logical_operations = logical_operations
        for condition in conditions:
            self.__dict__.update(condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the composite exit condition.

        Returns:
            str: String representation of the composite exit condition.
        """
        return f"{self.__class__.__name__}(conditions={self.conditions}, logical_operations='{self.logical_operations}')"

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        """
        Check if the composite exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
            manager (Optional[OptionBacktester]): The backtester instance managing the strategy. Defaults to None.

        Returns:
            bool: True if the composite exit condition is met, False otherwise.
        """
        if len(self.conditions) != len(self.logical_operations) + 1:
            raise ValueError(
                "The number of logical operations must be one less than the number of conditions."
            )

        results = [
            condition.should_exit(
                strategy, current_time, option_chain_df, manager=manager
            )
            for condition in self.conditions
        ]
        combined_result = results[0]

        for i, operation in enumerate(self.logical_operations):
            if operation == "AND":
                combined_result = combined_result and results[i + 1]
            elif operation == "OR":
                combined_result = combined_result or results[i + 1]
            else:
                raise ValueError("Logical operation must be 'AND' or 'OR'")

        if combined_result:
            if np.isnan(strategy.filter_pl):
                strategy.filter_pl = strategy.total_pl()

            if np.isnan(strategy.filter_return_percentage):
                strategy.filter_return_percentage = strategy.return_percentage()

        return combined_result

    def update(self, **kwargs):
        """
        Update the attributes of the composite exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        for condition in self.conditions:
            condition.update(**kwargs)
            self.__dict__.update(condition.__dict__)

    def __setattr__(self, key, value):
        """
        Override the __setattr__ method to update attributes in the conditions list.
        """
        super().__setattr__(key, value)
        for condition in self.conditions:
            if hasattr(condition, key):
                setattr(condition, key, value)
