import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.optopus.brokers.schwab.schwab_order import SchwabOptionOrder
from src.optopus.trades.option_spread import OptionStrategy, OptionLeg

class TestSchwabOptionOrder(unittest.TestCase):

    @patch('src.optopus.brokers.schwab.schwab_order.SchwabTrade')
    @patch('src.optopus.brokers.schwab.schwab_order.SchwabData')
    @patch('src.optopus.brokers.schwab.schwab_order.Order')
    def setUp(self, mock_order, mock_data, mock_trade):
        self.mock_trade = mock_trade
        self.mock_data = mock_data
        self.mock_order = mock_order

        # Mock account numbers
        self.mock_trade.return_value.get_account_numbers.return_value = [
            {"hashValue": "mock_account_hash"}
        ]

        # Mock option chain data
        self.option_chain_df = pd.DataFrame({
            "QUOTE_READTIME": ["2024-09-26 15:15:00"],
            "symbol": ["SPY"],
            "expiration": ["2024-11-15"],
            "strike": [300.0],
            "option_type": ["PUT"],
            "bid": [1.0],
            "ask": [1.5],
            "last": [1.25]
        })

        # Create a vertical spread strategy
        self.vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike=300.0,
            short_strike=305.0,
            expiration="2024-11-15",
            contracts=1,
            entry_time="2024-09-26 15:15:00",
            option_chain_df=self.option_chain_df
        )

        # Create a SchwabOptionOrder instance
        self.schwab_order = SchwabOptionOrder(
            option_strategy=self.vertical_spread,
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            token_file="token.json"
        )

    def test_generate_entry_payload(self):
        payload = self.schwab_order.generate_entry_payload()
        self.assertIsNotNone(payload)
        self.assertIn("symbol", payload)
        self.assertIn("expiration", payload)
        self.assertIn("longOptionType", payload)
        self.assertIn("longStrikePrice", payload)
        self.assertIn("shortOptionType", payload)
        self.assertIn("shortStrikePrice", payload)
        self.assertIn("quantity", payload)
        self.assertIn("price", payload)
        self.assertIn("duration", payload)
        self.assertIn("isEntry", payload)

    def test_submit_entry(self):
        with patch.object(self.schwab_order, 'place_order', return_value=["mock_order_id"]):
            result = self.schwab_order.submit_entry()
            self.assertTrue(result)
            self.assertEqual(self.schwab_order.order_id, "mock_order_id")
            self.assertIsNotNone(self.schwab_order.order_status)

    def test_generate_exit_payload(self):
        payload = self.schwab_order.generate_exit_payload()
        self.assertIsNotNone(payload)
        self.assertIn("symbol", payload)
        self.assertIn("expiration", payload)
        self.assertIn("longOptionType", payload)
        self.assertIn("longStrikePrice", payload)
        self.assertIn("shortOptionType", payload)
        self.assertIn("shortStrikePrice", payload)
        self.assertIn("quantity", payload)
        self.assertIn("price", payload)
        self.assertIn("duration", payload)
        self.assertIn("isEntry", payload)

    def test_submit_exit(self):
        with patch.object(self.schwab_order, 'place_order', return_value=["mock_order_id"]):
            result = self.schwab_order.submit_exit()
            self.assertTrue(result)
            self.assertEqual(self.schwab_order.order_id, "mock_order_id")
            self.assertIsNotNone(self.schwab_order.order_status)

    def test_update_order(self):
        self.schwab_order.update_order(self.option_chain_df)
        self.assertIsNotNone(self.schwab_order.current_time)
        self.assertIsNotNone(self.schwab_order.status)

    def test_update_order_status(self):
        with patch.object(self.schwab_order, 'get_order', return_value={"status": "FILLED"}):
            self.schwab_order.update_order_status()
            self.assertEqual(self.schwab_order.order_status, "FILLED")

    def test_cancel(self):
        with patch.object(self.schwab_order, 'cancel_order', return_value=True):
            self.schwab_order.cancel()
            self.assertEqual(self.schwab_order.order_status, "CANCELED")

    def test_modify(self):
        new_payload = {"new_key": "new_value"}
        with patch.object(self.schwab_order, 'modify_order', return_value=["mock_order_id"]):
            self.schwab_order.modify(new_payload)
            self.assertEqual(self.schwab_order.order_id, "mock_order_id")
            self.assertIsNotNone(self.schwab_order.order_status)

if __name__ == "__main__":
    unittest.main()
import os
import pandas as pd
import pytest
from src.optopus.brokers.schwab.schwab_order import SchwabOptionOrder
from src.optopus.trades.option_spread import OptionStrategy

# Load sample option chain data
entry_df = pd.read_parquet(
    "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-26 15-15.parquet"
)

# Create a vertical spread strategy
vertical_spread = OptionStrategy.create_vertical_spread(
    symbol="SPY",
    option_type="PUT",
    long_strike="-2",
    short_strike="ATM",
    expiration="2024-11-15",
    contracts=1,
    entry_time="2024-09-26 15:15:00",
    option_chain_df=entry_df,
)

# Create a SchwabOptionOrder instance
schwab_order = SchwabOptionOrder(
    client_id=os.getenv("SCHWAB_CLIENT_ID"),
    client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
    option_strategy=vertical_spread,
    token_file="token.json",
)

def test_schwab_order_details():
    assert schwab_order.symbol == "SPY"
    assert schwab_order.strategy_type == "Vertical Spread"
    assert schwab_order.legs[0].expiration == pd.to_datetime("2024-11-15")
    assert schwab_order.legs[0].strike == -2
    assert schwab_order.legs[1].strike == 400  # Assuming ATM strike is 400 for SPY
    assert schwab_order.contracts == 1
    assert schwab_order.entry_net_premium is not None
    assert schwab_order.order_status is None
    assert schwab_order.current_time is not None
    assert schwab_order.DIT is not None
    assert schwab_order.net_premium is not None

def test_schwab_order_update():
    schwab_order.update_order()
    assert schwab_order.current_time is not None
    assert schwab_order.entry_net_premium is not None
    assert schwab_order.order_status is not None
    assert schwab_order.DIT is not None
    assert schwab_order.net_premium is not None

def test_schwab_order_submit_entry():
    result = schwab_order.submit_entry()
    assert result is not None
    assert schwab_order.order_id is not None
    assert schwab_order.order_status == "FILLED"

def test_schwab_order_submit_exit():
    result = schwab_order.submit_exit()
    assert result is not None
    assert schwab_order.order_id is not None
    assert schwab_order.order_status == "FILLED"

if __name__ == "__main__":
    pytest.main(["-v", "test_schwab_order.py"])
