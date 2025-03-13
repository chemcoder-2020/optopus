from src.optopus.utils.ohlc_data_processor import DataProcessor
import unittest
import pandas as pd
from pathlib import Path

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        # Load test data
        self.intraday_data = "data/alpaca_SPY_15m_rebase.csv"
    
    def test_init_data_processor(self):
        dp = DataProcessor(self.intraday_data)
        if Path(self.intraday_data).exists():
            self.assertIsInstance(dp.intraday_data, pd.DataFrame)
        
    def test_prepare_historical_data(self):
        dp = DataProcessor(self.intraday_data)
        if Path(self.intraday_data).exists():
            self.assertIsInstance(dp.intraday_data, pd.DataFrame)
        
        # Test prepare_historical_data method at 2025-03-12 09:45:00
        self.assertEqual(dp.intraday_data[:"2025-03-12 09:45:00"][:-1].index[-1], pd.Timestamp("2025-03-12 09:30:00"))
        historical_data, monthly_data = dp.prepare_historical_data("2025-03-12 09:45:00", 100)
        self.assertEqual(historical_data.index[-1], pd.Timestamp("2025-03-12"))
        self.assertEqual(historical_data.index[-2], pd.Timestamp("2025-03-11"))
        self.assertEqual(monthly_data.index[-1], pd.Timestamp("2025-03-31"))
        self.assertEqual(monthly_data.index[-2], pd.Timestamp("2025-02-28"))
        self.assertEqual(len(historical_data), 500)
        self.assertEqual(historical_data["close"].iloc[-1], 560.6200)
        self.assertEqual(historical_data["open"].iloc[-1], 562.170)
        self.assertEqual(historical_data["high"].iloc[-1], 563.1100)
        self.assertEqual(historical_data["low"].iloc[-1], 560.060)
        self.assertEqual(historical_data["volume"].iloc[-1], 9510762.0)

        # Test prepare_historical_data method at 2025-03-12 10:15:00
        historical_data, monthly_data = dp.prepare_historical_data("2025-03-12 10:15:00", 100)
        self.assertEqual(historical_data.index[-1], pd.Timestamp("2025-03-12"))
        self.assertEqual(historical_data.index[-2], pd.Timestamp("2025-03-11"))
        self.assertEqual(monthly_data.index[-1], pd.Timestamp("2025-03-31"))
        self.assertEqual(monthly_data.index[-2], pd.Timestamp("2025-02-28"))
        self.assertEqual(len(historical_data), 500)
        self.assertEqual(historical_data["close"].iloc[-1], 559.3900)
        self.assertEqual(historical_data["open"].iloc[-1], 562.170)
        self.assertEqual(historical_data["high"].iloc[-1], 563.1100)
        self.assertEqual(historical_data["low"].iloc[-1], 557.980)
        self.assertEqual(historical_data["volume"].iloc[-1], 15383318)
