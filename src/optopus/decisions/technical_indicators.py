import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class TechnicalIndicators:
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate Average True Range (ATR)"""
        tr = pd.DataFrame(index=high.index)
        tr["h-l"] = high - low
        tr["h-pc"] = abs(high - close.shift(1))
        tr["l-pc"] = abs(low - close.shift(1))
        tr["tr"] = tr.max(axis=1)
        atr = tr["tr"].ewm(span=period, adjust=False).mean()
        return atr.iloc[-1]

    @staticmethod
    def check_median_trend(historical_data, short_lag=50, long_lag=200):
        """Check median forecast for upward trend"""
        median1 = np.median(historical_data["close"].iloc[-short_lag:].values)
        median2 = np.median(historical_data["close"].iloc[-long_lag:].values)
        return median1 > median2

    @staticmethod
    def check_rsi(historical_data, period=14, overbought=70, oversold=30) -> bool:
        """Check if RSI indicates oversold condition"""
        close_prices = historical_data["close"]
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] < oversold

    @staticmethod
    def check_linear_regression(historical_data, lag=3):
        """Check linear regression trend over specified lag period using daily data"""
        if len(historical_data) < lag:
            return False

        recent_data = historical_data["close"].iloc[-lag:]
        X = np.arange(lag).reshape(-1, 1)
        y = recent_data.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0] > 0
