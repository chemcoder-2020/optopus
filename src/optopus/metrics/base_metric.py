from abc import ABC, abstractmethod
import numpy as np
import scipy.stats
from typing import Any, Dict, Union
from ..utils.filters import HampelFilterNumpy
import pandas as pd

class BaseMetric(ABC):
    """Base class for all metric calculations"""
    
    @abstractmethod
    def calculate(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for the metric"""
        pass

    @staticmethod
    def safe_ratio(a: float, b: float) -> float:
        """Safe division handling zero denominator"""
        return a / b if b != 0 else 0.0
    
    @staticmethod
    def detect_outliers(series: Union[np.ndarray, pd.Series], window_size: int) -> None:
        pipe = HampelFilterNumpy(window_size=window_size, n_sigma=3, k=1.4826, max_iterations=5)
        series = pipe.fit_transform(series).flatten()
        return series
