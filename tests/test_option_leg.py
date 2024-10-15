import unittest
import pandas as pd
from src.optopus.trades.option_leg import OptionLeg

class TestOptionLeg(unittest.TestCase):

    def setUp(self):
        # Load test data from parquet files
        self.option_chain_df = pd.read_parquet('path_to_option_chain_data.parquet')
        self.entry_time = '2024-10-15 10:00:00'
        self.symbol = 'AAPL'
        self.option_type = 'CALL'
        self.strike = 150.0
        self.expiration = '2024-11-15'
        self.contracts = 1
        self.position_side = 'BUY'
        self.commission = 0.5

    def test_init(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        self.assertEqual(option_leg.symbol, self.symbol)
        self.assertEqual(option_leg.option_type, self.option_type)
        self.assertEqual(option_leg.strike, self.strike)
        self.assertEqual(option_leg.expiration, pd.to_datetime(self.expiration).tz_localize(None))
        self.assertEqual(option_leg.contracts, self.contracts)
        self.assertEqual(option_leg.entry_time, pd.to_datetime(self.entry_time).tz_localize(None))
        self.assertEqual(option_leg.position_side, self.position_side)
        self.assertEqual(option_leg.commission, self.commission)

    def test_update(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        new_current_time = '2024-10-16 10:00:00'
        option_leg.update(new_current_time, self.option_chain_df)
        self.assertEqual(option_leg.current_time, pd.to_datetime(new_current_time).tz_localize(None))

    def test_calculate_pl(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        pl = option_leg.calculate_pl()
        self.assertIsInstance(pl, float)

    def test_calculate_total_commission(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        total_commission = option_leg.calculate_total_commission()
        self.assertEqual(total_commission, self.commission * self.contracts * 2)

    def test_update_entry_price(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        new_entry_price = 155.0
        option_leg.update_entry_price(new_entry_price)
        self.assertEqual(option_leg.entry_price, new_entry_price)

    def test_conflicts_with(self):
        option_leg1 = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        option_leg2 = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        self.assertTrue(option_leg1.conflicts_with(option_leg2))

    def test_schwab_symbol(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.option_chain_df,
            position_side=self.position_side,
            commission=self.commission
        )
        expected_symbol = f"{self.symbol.ljust(6)}241115C00150000"
        self.assertEqual(option_leg.schwab_symbol, expected_symbol)

if __name__ == '__main__':
    unittest.main()
