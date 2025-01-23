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
            return {"max_drawdown": 0.0}

        peak = np.cummax(pl_curve)
        drawdown = peak - pl_curve
        max_drawdown_dollars = drawdown.max()
        max_drawdown_percentage = max_drawdown_dollars / allocation
        return {
            "max_drawdown_dollars": float(max_drawdown_dollars),
            "max_drawdown_percentage": float(max_drawdown_percentage),
        }
