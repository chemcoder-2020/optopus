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

