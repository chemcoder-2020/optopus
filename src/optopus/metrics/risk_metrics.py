import numpy as np
from scipy.stats import gaussian_kde
from .base_metric import BaseMetric


class SharpeRatio(BaseMetric):
    """Calculates Sharpe ratio from daily returns with risk-free rate adjustment"""

    def calculate(
        self, returns: np.ndarray, risk_free_rate: float = 0.0, window: int = 10
    ) -> dict:

        if returns.size < window + 1:
            return {"sharpe_ratio": 0.0}

        returns = returns.copy()
        returns = self.detect_outliers(returns, window_size=window)

        excess_returns = returns - risk_free_rate / 252
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        sharpe = self.safe_ratio(mean_return, std_return) * np.sqrt(252)
        return {"sharpe_ratio": float(sharpe)}


class RiskOfRuin(BaseMetric):
    """Calculates risk of ruin using Monte Carlo simulation"""

    def calculate(
        self,
        returns: np.ndarray,
        initial_balance: float,
        num_simulations: int = 20000,
        num_steps: int = 252,
        drawdown_threshold_pct: float = 0.25,
        distribution: str = "histogram",
        window_size: int = 10,
    ) -> dict:
        """
        Args:
            returns (np.ndarray): Array of trade returns
            initial_balance (float): Initial capital balance
            num_simulations (int): Number of Monte Carlo simulations
            num_steps (int): Number of steps in each simulation
            drawdown_threshold_pct (float): Drawdown threshold percentage
            distribution (str): Distribution type ("normal", "kde", "histogram")

        Returns:
            dict: Dictionary with risk_of_ruin percentage
        """
        # Apply rolling median to returns
        if returns.size < window_size + 1:
            return {"risk_of_ruin": 0.0}

        returns = returns.copy()
        returns = self.detect_outliers(returns, window_size=window_size)

        returns = returns / initial_balance

        if distribution == "normal":
            random_returns = np.random.normal(
                np.mean(returns), np.std(returns), size=(num_simulations, num_steps)
            )
        elif distribution == "kde":
            kde = gaussian_kde(returns, bw_method="scott")
            samples = kde.resample(size=(num_simulations * num_steps))
            random_returns = samples.T
        elif distribution == "histogram":
            random_returns = np.random.choice(
                returns, size=(num_simulations, num_steps)
            )
        else:
            raise ValueError("Unsupported distribution type")

        balances = initial_balance + np.cumsum(random_returns * initial_balance, axis=1)
        peak_balances = np.maximum.accumulate(balances, axis=1)
        drawdown_thresholds = peak_balances - drawdown_threshold_pct * initial_balance
        ruin_count = np.sum(np.any(balances <= drawdown_thresholds, axis=1))

        return {"risk_of_ruin": float(ruin_count / num_simulations)}


class MaxDrawdown(BaseMetric):
    """Calculates maximum drawdown from cumulative returns"""

    def calculate(
        self, pl_curve: np.ndarray, allocation: float, window: int = 10
    ) -> dict:

        # Apply rolling median
        window = min(window, pl_curve.size)  # Adjust window size if needed
        if pl_curve.size < window + 1:
            return {"max_drawdown_dollars": 0.0, "max_drawdown_percentage": 0.0}
        pl_curve = pl_curve.copy()
        pl_curve = self.detect_outliers(pl_curve, window_size=window)

        # Calculate running maximum
        peak = np.maximum.accumulate(pl_curve)
        # Calculate drawdown from peak
        drawdown = peak - pl_curve

        # Find maximum drawdown
        max_drawdown_dollars = drawdown.max()
        max_drawdown_percentage = max_drawdown_dollars / allocation
        max_drawdown_percentage_from_peak = np.nanmax(drawdown / (peak+allocation))

        return {
            "max_drawdown_dollars": float(max_drawdown_dollars),
            "max_drawdown_percentage": float(max_drawdown_percentage),
            "max_drawdown_percentage_from_peak": float(
                max_drawdown_percentage_from_peak
            ),
        }
