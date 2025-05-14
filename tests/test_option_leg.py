import unittest
import pandas as pd
import datetime
from src.optopus.trades.option_leg import OptionLeg, calculate_dte


class TestOptionLeg(unittest.TestCase):

    def setUp(self):
        # Load test data from parquet files
        self.entry_df = pd.read_parquet("data/SPY_2024-10-15 10-00.parquet")
        self.update_df = pd.read_parquet("data/SPY_2024-10-15 15-45.parquet")
        self.entry_time = "2024-10-15 10:00:00"
        self.symbol = "SPY"
        self.option_type = "CALL"
        self.strike = 585
        self.expiration = "2024-12-20"
        self.contracts = 1
        self.position_side = "SELL"
        self.commission = 0.5

    def test_init(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        self.assertEqual(option_leg.symbol, self.symbol)
        self.assertEqual(option_leg.option_type, self.option_type)
        self.assertEqual(option_leg.strike, self.strike)
        self.assertEqual(
            option_leg.expiration, pd.to_datetime(self.expiration).tz_localize(None)
        )
        self.assertEqual(option_leg.contracts, self.contracts)
        self.assertEqual(
            option_leg.entry_time, pd.to_datetime(self.entry_time).tz_localize(None)
        )
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
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        self.assertAlmostEqual(option_leg.underlying_last, 584.33, places=2)
        self.assertAlmostEqual(option_leg.current_ask, 17.17, places=2)
        self.assertAlmostEqual(option_leg.current_bid, 16.94, places=2)
        self.assertAlmostEqual(option_leg.current_mark, 17.06, places=2)
        self.assertAlmostEqual(option_leg.current_last, 17.0, places=2)
        self.assertAlmostEqual(option_leg.current_delta, 0.551, places=3)
        new_current_time = "2024-10-15 15:45:00"
        option_leg.update(new_current_time, self.update_df)
        self.assertEqual(
            option_leg.current_time, pd.to_datetime(new_current_time).tz_localize(None)
        )
        self.assertAlmostEqual(option_leg.underlying_last, 579.27, places=2)
        self.assertAlmostEqual(option_leg.current_ask, 14.43, places=1)
        self.assertAlmostEqual(option_leg.current_bid, 14.4, places=1)
        self.assertAlmostEqual(option_leg.current_mark, 14.42, places=1)
        self.assertAlmostEqual(option_leg.current_last, 14.52, places=1)
        self.assertAlmostEqual(option_leg.current_delta, 0.499, places=3)

        self.assertAlmostEqual(option_leg.underlying_diff, 579.27 - 584.33, places=2)
        self.assertAlmostEqual(option_leg.price_diff, 14.42 - 17.06, places=2)
        self.assertAlmostEqual(
            option_leg.pl,
            -1 * (14.42 - 17.06) * self.contracts * 100
            - option_leg.calculate_total_commission(),
            places=2,
        )

    def test_calculate_pl(self):
        option_leg = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
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
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
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
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        new_current_time = "2024-10-15 15:45:00"
        option_leg.update(new_current_time, self.update_df)
        old_entry_price = option_leg.entry_price
        old_price_diff = option_leg.price_diff
        old_pl = option_leg.calculate_pl()
        new_entry_price = old_entry_price + 0.1

        expected_new_price_diff = old_price_diff - 0.1
        expected_new_pl = (
            -1 * expected_new_price_diff * option_leg.contracts * 100
            - option_leg.calculate_total_commission()
        )

        option_leg.update_entry_price(new_entry_price)
        new_pl = option_leg.calculate_pl()
        self.assertAlmostEqual(option_leg.entry_price, new_entry_price, places=3)
        self.assertAlmostEqual(option_leg.price_diff, expected_new_price_diff, places=3)
        self.assertAlmostEqual(option_leg.pl, expected_new_pl, places=3)

    def test_conflicts_with(self):
        option_leg1 = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        option_leg2 = OptionLeg(
            symbol=self.symbol,
            option_type=self.option_type,
            strike=self.strike,
            expiration=self.expiration,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        self.assertTrue(option_leg1.conflicts_with(option_leg2))

    def test_from_delta_and_dte_atm(self):
        target_delta = "ATM"
        target_dte = 60
        option_leg = OptionLeg.from_delta_and_dte(
            symbol=self.symbol,
            option_type=self.option_type,
            target_delta=target_delta,
            target_dte=target_dte,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        self.assertEqual(option_leg.symbol, self.symbol)
        self.assertEqual(option_leg.strike, 585)
        self.assertEqual(option_leg.expiration, pd.to_datetime(self.expiration))
        self.assertEqual(option_leg.option_type, self.option_type)
        self.assertEqual(option_leg.contracts, self.contracts)
        self.assertEqual(option_leg.position_side, self.position_side)
        self.assertEqual(option_leg.commission, self.commission)

    def test_from_delta_and_dte_specific_delta(self):
        target_delta = -0.3
        target_dte = 60
        option_leg = OptionLeg.from_delta_and_dte(
            symbol=self.symbol,
            option_type="PUT",
            target_delta=target_delta,
            target_dte=target_dte,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            commission=self.commission,
        )
        self.assertEqual(option_leg.symbol, self.symbol)
        self.assertEqual(option_leg.strike, 567)
        self.assertEqual(option_leg.expiration, pd.to_datetime(self.expiration))
        self.assertEqual(option_leg.option_type, "PUT")
        self.assertEqual(option_leg.contracts, self.contracts)
        self.assertEqual(option_leg.position_side, self.position_side)
        self.assertEqual(option_leg.commission, self.commission)

    def test_from_delta_and_dte_relative_strike(self):
        target_delta = -2
        target_dte = 60
        reference_strike = 580
        option_leg = OptionLeg.from_delta_and_dte(
            symbol=self.symbol,
            option_type="PUT",
            target_delta=target_delta,
            target_dte=target_dte,
            contracts=self.contracts,
            entry_time=self.entry_time,
            option_chain_df=self.entry_df,
            position_side=self.position_side,
            reference_strike=reference_strike,
            commission=self.commission,
        )
        self.assertEqual(option_leg.symbol, self.symbol)
        self.assertEqual(option_leg.strike, reference_strike + target_delta)
        self.assertEqual(option_leg.expiration, pd.to_datetime(self.expiration))
        self.assertEqual(option_leg.option_type, "PUT")
        self.assertEqual(option_leg.contracts, self.contracts)
        self.assertEqual(option_leg.position_side, self.position_side)
        self.assertEqual(option_leg.commission, self.commission)

    def test_calculate_dte_data_types(self):
        expiration_date_str = "2024-12-20"
        current_date_str = "2024-10-15"
        expiration_date_ts = pd.to_datetime(expiration_date_str)
        current_date_ts = pd.to_datetime(current_date_str)
        expiration_date_dt = datetime.datetime.strptime(expiration_date_str, "%Y-%m-%d")
        current_date_dt = datetime.datetime.strptime(current_date_str, "%Y-%m-%d")

        expected_dte = calculate_dte(expiration_date_str, current_date_str)

        self.assertEqual(
            calculate_dte(expiration_date_str, current_date_ts), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_str, current_date_dt), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_ts, current_date_str), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_ts, current_date_ts), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_ts, current_date_dt), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_dt, current_date_str), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_dt, current_date_ts), expected_dte
        )
        self.assertEqual(
            calculate_dte(expiration_date_dt, current_date_dt), expected_dte
        )


if __name__ == "__main__":
    unittest.main()
