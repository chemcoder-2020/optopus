from .base import Preprocessor
from ...utils.filters import HampelFilterNumpy, Filter
from typing import Union, Type
from loguru import logger
import importlib


class PremiumFilter(Preprocessor):
    def __init__(
        self,
        filter_method: Union[Filter, Type[Filter], str] = HampelFilterNumpy,
        **kwargs,
    ):
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

    def preprocess(self, strategy, manager):
        """
        Check if current premium's return percentage is an outlier. Updates strategy's filtered metrics with the cleaned values.

        Args:
            strategy (OptionStrategy): The option strategy to check
        """
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2  # if bid != 0 else ask
        manager.context["premiums"].append(mark)
        if len(manager.context["premiums"]) > self.window_size + 1:
            manager.context["premiums"].pop(0)

        if len(manager.context["premiums"]) < self.window_size + 1:
            logger.debug(
                f"Manager's premiums context has less than {self.window_size + 1} values, skipping filter - assume valid data"
            )
            return mark

        filtered_returns = self.premium_filter.fit_transform(
            manager.context["premiums"]
        )

        return filtered_returns[-1]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.filter_method.__name__}: {self.kwargs})"
        )
