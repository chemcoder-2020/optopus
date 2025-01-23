import numpy as np
import pandas as pd
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

class CAGR(BaseMetric):
    """Calculates Compound Annual Growth Rate (CAGR)"""
    
    def calculate(self, initial_value: float, final_value: float, 
                 start_time: pd.Timestamp, end_time: pd.Timestamp) -> dict:
        """
        Args:
            initial_value: Starting portfolio value
            final_value: Ending portfolio value
            start_time: Start timestamp of the period
            end_time: End timestamp of the period
            
        Returns:
            Dictionary with cagr percentage
        """
        if initial_value <= 0:
            return {"cagr": 0.0}
            
        delta = end_time - start_time
        years = delta.days / 365.25
        
        try:
            cagr = (final_value / initial_value) ** (1 / years) - 1
        except ZeroDivisionError:
            cagr = 0.0
            
        return {"cagr": float(cagr)}

class MonthlyReturn(BaseMetric):
    """Calculates average monthly profit/loss from performance data"""
    
    def calculate(self, closed_pl_series: pd.Series) -> dict:
        """
        Args:
            closed_pl_series (pd.Series): Series of closed P/L values with datetime index
            
        Returns:
            Dictionary with average monthly P/L
        """
        if closed_pl_series.empty:
            return {"avg_monthly_pl": 0.0}
            
        # Resample to monthly and calculate changes
        monthly_pl = closed_pl_series.resample("M").last().diff().dropna()
        return {"avg_monthly_pl": float(monthly_pl.mean())}
