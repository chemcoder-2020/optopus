from abc import ABC, abstractmethod
import numpy as np
import scipy.stats
from typing import Any, Dict

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
