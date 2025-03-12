from .base import Preprocessor
from ...utils.filters import HampelFilterNumpy, Filter
from typing import Union, Type
from loguru import logger
import importlib
import numpy as np


class RORFilter(Preprocessor):
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

        self.ror_filter = self.filter_method(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def preprocess(self, strategy, manager):
        """
        Check if current return_over_risk is an outlier. Updates strategy's filtered metrics with the cleaned values.

        Args:
            strategy (OptionStrategy): The option strategy to check
        """
        return_over_risk = strategy.return_over_risk()
        manager.context["RoRs"].append(return_over_risk)

        filtered_rors = self.ror_filter.fit_transform(
            manager.context["RoRs"]
        )

        return filtered_rors[-1]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.filter_method.__name__}: {self.kwargs})"
        )
