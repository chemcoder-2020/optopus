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
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.trend import TrendForecaster
from sklearn.linear_model import Ridge
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
    def check_ML_trend(monthly_data, classifier="random_forest"):
        """
        Predict next month's direction using a machine learning classifier.

        Args:
            monthly_data: Pandas Series of monthly prices
            classifier: Classifier to use ('logistic', 'svm', 'random_forest', 'gradient_boosting')
                      Defaults to 'random_forest'

        Returns:
            bool: True if classifier predicts upward trend
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier

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

        # Initialize selected classifier
        if classifier == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression())
        elif classifier == "svm":
            model = make_pipeline(StandardScaler(), SVC(probability=True))
        elif classifier == "gradient_boosting":
            model = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=400))
        elif classifier == "gaussian_process":
            model = make_pipeline(StandardScaler(), GaussianProcessClassifier())
        elif classifier == "mlp":
            model = make_pipeline(StandardScaler(), MLPClassifier(max_iter=100))
        elif classifier == "knn":
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(2))
        else:  # default to random_forest
            model = make_pipeline(StandardScaler(), RandomForestClassifier())

        # Train and predict
        model.fit(X, y)
        return model.predict(latest_data)[0] == 1

    @staticmethod
    def check_seasonality_oscillator(
        monthly_data: pd.Series, 
        threshold: float = 1.0,
        seasonal_period: int = 12,
        lags: int = 3
    ) -> bool:
        """
        Detect strong seasonal patterns using STL decomposition and transformation pipeline.
        
        Args:
            monthly_data: Pandas Series of monthly prices
            threshold: Z-score threshold to consider significant (default: 1.0)
            seasonal_period: Number of periods in seasonal cycle (default: 12)
            lags: Number of lags for differencing (default: 3)
            
        Returns:
            bool: True if latest seasonal component exceeds threshold
        """
        try:
            # Create transformation pipeline
            pipe = (
                LogTransformer() *
                Detrender(TrendForecaster(Ridge())) *
                Differencer(lags=lags)
            )
            
            # Fit and transform pipeline
            oscillator = pipe.fit_transform(monthly_data)
            
            # Calculate z-score of latest value
            zscore = (oscillator[-1] - oscillator.mean()) / oscillator.std()
            logger.info(f"Seasonality oscillator z-score: {zscore:.2f}")
            
            return zscore > threshold
            
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {str(e)}")
            return False
