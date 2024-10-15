import unittest
from src.optopus.trades.option_manager import OptionBacktester, Config
from src.optopus.trades.option_spread import OptionStrategy
import pandas as pd
from datetime import datetime

class TestOptionBacktester(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            initial_capital=10000,
            max_positions=5,
            max_positions_per_day=2,
            max_positions_per_week=5,
            position_size=0.1,
            ror_threshold=0.05,
            gain_reinvesting=False,
        )
        self.backtester = OptionBacktester(self.config)
        self.entry_df = pd.DataFrame({
            "UNDERLYING_LAST": [400],
            "expiration": ["2024-12-20"],
            "strike": [400],
            "option_type": ["CALL"],
            "entry_time": ["2024-09-06 15:30:00"],
        })
        self.update_df = pd.DataFrame({
            "UNDERLYING_LAST": [410],
            "expiration": ["2024-12-20"],
            "strike": [400],
            "option_type": ["CALL"],
            "entry_time": ["2024-09-06 15:45:00"],
        })

    def test_initialization(self):
        self.assertEqual(self.backtester.capital, 10000)
        self.assertEqual(self.backtester.config.max_positions, 5)

    def test_add_spread(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(self.backtester.add_spread(spread))
        self.assertEqual(len(self.backtester.active_trades), 1)

    def test_add_conflicting_spread(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(self.backtester.add_spread(spread))
        conflicting_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertFalse(self.backtester.add_spread(conflicting_spread))
        self.assertEqual(len(self.backtester.active_trades), 1)

    def test_update_backtester(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(self.backtester.add_spread(spread))
        initial_capital = self.backtester.capital
        self.backtester.update("2024-09-06 15:45:00", self.update_df)
        self.assertEqual(self.backtester.trades_entered_today, 1)
        self.assertEqual(self.backtester.trades_entered_this_week, 1)
        self.assertEqual(self.backtester.capital, initial_capital - spread.get_required_capital())

    def test_update_with_position_closing(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            profit_target=1,
        )
        self.assertTrue(self.backtester.add_spread(spread))
        self.backtester.update("2024-09-06 15:45:00", self.update_df)
        self.assertEqual(len(self.backtester.active_trades), 0)

    def test_maximum_positions(self):
        for i in range(2):
            spread = OptionStrategy.create_vertical_spread(
                symbol="SPY",
                option_type="CALL",
                long_strike=400 + i,
                short_strike=390 + i,
                expiration="2024-12-20",
                contracts=1,
                entry_time="2024-09-06 15:30:00",
                option_chain_df=self.entry_df,
            )
            self.assertTrue(self.backtester.add_spread(spread))
        self.assertEqual(len(self.backtester.active_trades), 2)
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=402,
            short_strike=392,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertFalse(self.backtester.add_spread(spread))
        self.assertEqual(len(self.backtester.active_trades), 2)

    def test_profit_loss_calculation(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(self.backtester.add_spread(spread))
        self.backtester.update("2024-09-06 15:45:00", self.update_df)
        total_pl = self.backtester.get_total_pl()
        self.assertIsInstance(total_pl, (float, int))
        closed_pl = self.backtester.get_closed_pl()
        self.assertIsInstance(closed_pl, (float, int))
        self.assertEqual(closed_pl, 0)

    def test_adjusting_contracts_to_fit_position_size(self):
        spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=400,
            short_strike=390,
            expiration="2024-12-20",
            contracts=1000,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        max_capital = self.backtester.allocation * self.backtester.config.position_size
        expected_contracts = int(max_capital // spread.get_required_capital_per_contract())
        expected_required_capital = (
            spread.get_required_capital_per_contract() * expected_contracts
        )
        self.assertTrue(self.backtester.add_spread(spread))
        self.assertEqual(spread.contracts, expected_contracts)
        self.assertEqual(spread.get_required_capital(), expected_required_capital)

if __name__ == "__main__":
    unittest.main()
