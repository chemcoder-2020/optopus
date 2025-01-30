import warnings
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA,
    SeasonalExponentialSmoothingOptimized,
    RandomWalkWithDrift,
    AutoARIMA,
    AutoCES,
)
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from loguru import logger


class ForecastModels:
    @staticmethod
    def check_arima_trend(monthly_data, current_price, freq="M"):
        """Check ARIMA forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = StatsForecast(
                models=[ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1))],
                freq=freq,
            )
            sf_df = pd.DataFrame(
                {
                    "ds": list(monthly_data.index),
                    "y": list(monthly_data.values),
                    "unique_id": "1",
                }
            )
            forecaster.fit(sf_df)
            predictions = forecaster.predict(h=1)["ARIMA"]
            return predictions.iloc[-1] > current_price

    @staticmethod
    def check_autoarima_trend(monthly_data, current_price, freq="M"):
        """Check AutoARIMA forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = StatsForecast(
                models=[AutoARIMA()],
                freq=freq,
            )
            sf_df = pd.DataFrame(
                {
                    "ds": list(monthly_data.index),
                    "y": list(monthly_data.values),
                    "unique_id": "1",
                }
            )
            forecaster.fit(sf_df)
            predictions = forecaster.predict(h=1)["AutoARIMA"]
            return predictions.iloc[-1] > current_price

    @staticmethod
    def check_autoces_trend(monthly_data, current_price, freq="M"):
        """Check AutoCES forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = StatsForecast(
                models=[AutoCES()],
                freq=freq,
            )
            sf_df = pd.DataFrame(
                {
                    "ds": list(monthly_data.index),
                    "y": list(monthly_data.values),
                    "unique_id": "1",
                }
            )
            forecaster.fit(sf_df)
            predictions = forecaster.predict(h=1)["CES"]
            return predictions.iloc[-1] > current_price

    @staticmethod
    def check_nbeats_trend(monthly_data):
        """Check NBEATS forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = NeuralForecast(
                models=[
                    NBEATS(
                        input_size=4,
                        h=2,
                        max_steps=100,
                        enable_progress_bar=False,
                        logger=False,
                        enable_checkpointing=False,
                    )
                ],
                freq="M",
            )
            nf_df = pd.DataFrame(
                {
                    "ds": list(monthly_data.index),
                    "y": list(monthly_data),
                    "unique_id": "1",
                }
            )
            forecaster.fit(nf_df)
            predictions = forecaster.predict()[forecaster.models[0].__str__()]
            return predictions.iloc[1] - predictions.iloc[0] > 0

    @staticmethod
    def check_arima_trend_daily(daily_data, current_price):
        """Check ARIMA forecast for upward trend using daily data"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = StatsForecast(
                models=[ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1))],
                freq="D",
            )
            sf_df = pd.DataFrame(
                {"ds": list(daily_data.index), "y": list(daily_data), "unique_id": "1"}
            )
            forecaster.fit(sf_df)
            predictions = forecaster.predict(h=1)["ARIMA"]
            return predictions.iloc[-1] > current_price

    @staticmethod
    def check_RWD_trend(monthly_data, current_price):
        """Check Random Walk with Drift forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = StatsForecast(
                models=[RandomWalkWithDrift()],
                freq="M",
            )
            sf_df = pd.DataFrame(
                {
                    "ds": list(monthly_data.index),
                    "y": list(monthly_data),
                    "unique_id": "1",
                }
            )
            forecaster.fit(sf_df)
            predictions = forecaster.predict(h=1)["RWD"]
            return predictions.iloc[-1] > current_price

    @staticmethod
    def check_SES_trend(monthly_data, current_price):
        """Check Seasonal Exponential Smoothing forecast for upward trend"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            forecaster = SeasonalExponentialSmoothingOptimized(season_length=1)
            forecaster.fit(y=monthly_data.values)
            predictions = forecaster.predict(h=1)["mean"]
            return predictions[-1] > current_price

    @staticmethod
    def check_ML_trend(monthly_data, classifiers=None):
        """
        Predict next month's direction using multiple machine learning classifiers.

        Args:
            monthly_data: Pandas Series of monthly prices
            classifiers: List of classifier names to use ('logistic', 'svm', 'random_forest')
                        Defaults to using all available classifiers

        Returns:
            bool: True if majority of classifiers predict upward trend
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Copy data
        monthly_data = monthly_data.copy().to_frame()
        logger.info(f"Monthly data: {monthly_data}")
        monthly_data.columns = ['close']
        latest_data = monthly_data[-1:]
        logger.info(f"Latest data: {latest_data}")
        monthly_data["target"] = (
            monthly_data.shift(-1) > monthly_data
        ).astype(int)
        monthly_data = monthly_data.dropna()

        # Create features and target
        X = monthly_data.drop(columns=["target"])
        y = monthly_data["target"]

        # Initialize classifiers
        if classifiers is None:
            classifiers = ["logistic", "svm", "random_forest", "gradient_boosting"]

        models = []
        if "logistic" in classifiers:
            models.append(make_pipeline(StandardScaler(), LogisticRegression()))
        if "svm" in classifiers:
            models.append(make_pipeline(StandardScaler(), SVC(probability=True)))
        if "random_forest" in classifiers:
            models.append(make_pipeline(StandardScaler(), RandomForestClassifier()))
        if "gradient_boosting" in classifiers:
            models.append(make_pipeline(StandardScaler(), GradientBoostingClassifier()))

        predictions = []
        for model in models:
            model.fit(X, y)
            pred = model.predict(latest_data)[0]
            predictions.append(pred)

        # Return majority vote
        return sum(predictions) > len(predictions) / 2
