import numpy as np
import scipy.stats
from .base_metric import BaseMetric


class SharpeRatio(BaseMetric):
    """Calculates Sharpe ratio from daily returns with risk-free rate adjustment"""

    def calculate(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> dict:
        if returns.size < 2:
            return {"sharpe_ratio": 0.0}

        excess_returns = returns - risk_free_rate / 252
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        sharpe = self.safe_ratio(mean_return, std_return) * np.sqrt(252)
        return {"sharpe_ratio": float(sharpe)}


class MaxDrawdown(BaseMetric):
    """Calculates maximum drawdown from cumulative returns"""
    
    def calculate(self, pl_curve: np.ndarray, allocation: float) -> dict:
        if pl_curve.size == 0:
            return {
                "max_drawdown_dollars": 0.0,
                "max_drawdown_percentage": 0.0
            }

        # Calculate running maximum
        peak = np.maximum.accumulate(pl_curve)
        # Calculate drawdown from peak
        drawdown = (peak - pl_curve) / np.where(peak == 0, 1, peak)  # Handle zero peak
        
        max_dd_idx = np.argmax(drawdown)
        max_drawdown_dollars = peak[max_dd_idx] - pl_curve[max_dd_idx]
        max_drawdown_percentage = drawdown[max_dd_idx]
        
        return {
            "max_drawdown_dollars": float(max_drawdown_dollars),
            "max_drawdown_percentage": float(max_drawdown_percentage),
        }
