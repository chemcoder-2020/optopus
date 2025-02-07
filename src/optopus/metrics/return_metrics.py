import numpy as np
import pandas as pd
from .base_metric import BaseMetric


class TotalReturn(BaseMetric):
    """Calculates total return percentage from a series of returns"""

    def calculate(self, returns: np.ndarray, window: int = 10) -> dict:

        if returns.size < window + 1:
            return {"total_return": 0.0}

        returns = returns.copy()
        returns = self.detect_outliers(returns, window_size=window)

        total_return = np.exp(np.sum(returns)) - 1
        return {"total_return": float(total_return)}


class AnnualizedReturn(BaseMetric):
    """Calculates annualized return from daily returns"""

    def calculate(self, returns: np.ndarray, window: int = 10) -> dict:
        if returns.size < window + 1:
            return {"annualized_return": 0.0}
        returns = returns.copy()
        returns = self.detect_outliers(returns, window_size=window)

        cumulative_return = np.exp(np.sum(returns))
        days = len(returns)
        annualized_return = cumulative_return ** (252 / days) - 1
        return {"annualized_return": float(annualized_return)}


class CAGR(BaseMetric):
    """Calculates Compound Annual Growth Rate (CAGR)"""

    def calculate(
        self,
        initial_value: float,
        final_value: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> dict:
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

    def calculate(
        self, closed_pl_series: pd.Series, non_zero_only: bool = False, window: int = 10
    ) -> dict:
        """
        Args:
            closed_pl_series (pd.Series): Series of closed P/L values with datetime index
            non_zero_only (bool): If True, average only months with non-zero P/L

        Returns:
            Dictionary with average monthly P/L
        """
        if closed_pl_series.size < window + 1:
            return {"avg_monthly_pl": 0.0}

        closed_pl_series = closed_pl_series.copy()
        closed_pl_series = self.detect_outliers(closed_pl_series, window_size=window)

        # Resample to monthly and calculate changes
        monthly_pl = closed_pl_series.resample("M").last().diff().dropna()

        if non_zero_only:
            monthly_pl = monthly_pl[monthly_pl != 0]

        return {
            "avg_monthly_pl": float(monthly_pl.mean()) if not monthly_pl.empty else 0.0
        }


class PositiveMonthlyProbability(BaseMetric):
    """Calculates probability of positive monthly P/L"""

    def calculate(self, pl_series: pd.Series, window: int = 10) -> dict:
        """
        Args:
            pl_series (pd.Series): Series of P/L values with datetime index

        Returns:
            Dictionary with probability of positive months
        """
        if pl_series.size < window + 1:
            return {"positive_monthly_probability": 0.0}

        pl_series = pl_series.copy()
        pl_series = self.detect_outliers(pl_series, window_size=window)
        monthly_pl = pl_series.resample("M").last().diff().dropna()
        positive_months = monthly_pl[monthly_pl > 0]
        total_months = monthly_pl[monthly_pl != 0]
        prob = len(positive_months) / len(total_months) if len(total_months) > 0 else 0
        return {"positive_monthly_probability": float(prob)}
