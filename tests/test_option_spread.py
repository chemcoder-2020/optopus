import unittest
import pandas as pd
from src.optopus.trades.option_spread import OptionStrategy
from src.optopus.trades.option_leg import OptionLeg
from src.optopus.trades.exit_conditions import DefaultExitCondition, CompositeExitCondition, TimeBasedCondition, ProfitTargetCondition, StopLossCondition
from loguru import logger

class TestOptionStrategy(unittest.TestCase):

    def setUp(self):
        self.entry_df = pd.read_parquet("data/SPY_2024-09-06 15-30.parquet")
        self.update_df = pd.read_parquet("data/SPY_2024-09-06 15-45.parquet")
        self.update_df2 = pd.read_parquet("data/SPY_2024-09-09 09-45.parquet")

    def test_vertical_spread_creation(self):
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(vertical_spread.legs), 2)
        self.assertEqual(vertical_spread.strategy_type, "Vertical Spread")

    def test_iron_condor_creation(self):
        iron_condor = OptionStrategy.create_iron_condor(
            symbol="SPY",
            put_long_strike="-5",
            put_short_strike="-0.3",
            call_short_strike="+0.3",
            call_long_strike="+5",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(iron_condor.legs), 4)
        self.assertEqual(iron_condor.strategy_type, "Iron Condor")

    def test_straddle_creation(self):
        straddle = OptionStrategy.create_straddle(
            symbol="SPY",
            strike="ATM",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(straddle.legs), 2)
        self.assertEqual(straddle.strategy_type, "Straddle")

    def test_butterfly_creation(self):
        butterfly = OptionStrategy.create_butterfly(
            symbol="SPY",
            option_type="CALL",
            lower_strike=540,
            middle_strike=550,
            upper_strike=560,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(butterfly.legs), 3)
        self.assertEqual(butterfly.strategy_type, "Butterfly")

    def test_naked_call_creation(self):
        naked_call = OptionStrategy.create_naked_call(
            symbol="SPY",
            strike=550,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(naked_call.legs), 1)
        self.assertEqual(naked_call.strategy_type, "Naked Call")

    def test_naked_put_creation(self):
        naked_put = OptionStrategy.create_naked_put(
            symbol="SPY",
            strike=540,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(len(naked_put.legs), 1)
        self.assertEqual(naked_put.strategy_type, "Naked Put")

    def test_strategy_update(self):
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        vertical_spread.update("2024-09-06 15:30:00", self.entry_df)
        self.assertTrue(
            vertical_spread.total_pl()
            == 0 - vertical_spread.calculate_total_commission()
        )
        self.assertEqual(
            vertical_spread.current_time, pd.Timestamp("2024-09-06 15:30:00")
        )
        vertical_spread.update("2024-09-06 15:45:00", self.update_df)
        self.assertTrue(
            vertical_spread.total_pl() + vertical_spread.calculate_total_commission()
            > 0
        )
        self.assertEqual(
            vertical_spread.current_time, pd.Timestamp("2024-09-06 15:45:00")
        )
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertTrue(
            vertical_spread.total_pl() + vertical_spread.calculate_total_commission()
            < 0
        )
        self.assertEqual(
            vertical_spread.current_time, pd.Timestamp("2024-09-09 09:45:00")
        )

    def test_conflict_vertical_spread_same_legs(self):
        spread1 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(spread1.conflicts_with(spread2))

    def test_conflict_vertical_spread_one_same_leg(self):
        spread1 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=555,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertTrue(spread1.conflicts_with(spread2))

    def test_conflict_vertical_spread_two_different_legs(self):
        spread1 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=561,
            short_strike=551,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertFalse(spread1.conflicts_with(spread2))

    def test_conflict_vertical_spread_different_option_type(self):
        spread1 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertFalse(spread1.conflicts_with(spread2))

    def test_conflict_vertical_spread_different_expiration(self):
        spread1 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-11-15",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertFalse(spread1.conflicts_with(spread2))

    def test_required_capital(self):
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        capital = vertical_spread.get_required_capital()
        self.assertGreater(capital, 0)

    def test_dit_calculation(self):
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(vertical_spread.DIT, 0)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertEqual(vertical_spread.DIT, 3)

    def test_close_strategy(self):
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        vertical_spread.close_strategy("2024-09-06 15:45:00", self.update_df)
        self.assertEqual(vertical_spread.status, "CLOSED")

    def test_won_attribute(self):
        stop_loss_condition = StopLossCondition(stop_loss=10)
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=stop_loss_condition,
        )
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertFalse(vertical_spread.won)


if __name__ == "__main__":
    unittest.main()
    def test_stop_loss_condition(self):
        stop_loss_condition = StopLossCondition(stop_loss=10)
        vertical_spread = OptionStrategy.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=stop_loss_condition,
        )
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertFalse(vertical_spread.won)
