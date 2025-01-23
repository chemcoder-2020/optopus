import numpy as np
from .base_metric import BaseMetric

class TotalReturn(BaseMetric):
    """Calculates total return percentage from a series of returns"""
    
    def calculate(self, returns: np.ndarray) -> dict:
        if returns.size == 0:
            return {"total_return": 0.0}
            
        total_return = np.exp(np.sum(returns)) - 1
        return {"total_return": float(total_return)}

class AnnualizedReturn(BaseMetric):
    """Calculates annualized return from daily returns"""
    
    def calculate(self, returns: np.ndarray) -> dict:
        if returns.size == 0:
            return {"annualized_return": 0.0}
            
        cumulative_return = np.exp(np.sum(returns))
        days = len(returns)
        annualized_return = cumulative_return ** (252 / days) - 1
        return {"annualized_return": float(annualized_return)}
