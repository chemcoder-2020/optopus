import unittest
import pandas as pd
from src.optopus.trades.option_spread import OptionStrategy
from src.optopus.trades.strategies.vertical_spread import VerticalSpread
from src.optopus.trades.strategies.iron_condor import IronCondor
from src.optopus.trades.strategies.straddle import Straddle
from src.optopus.trades.strategies.iron_butterfly import IronButterfly
from src.optopus.trades.strategies.naked_put import NakedPut
from src.optopus.trades.strategies.naked_call import NakedCall
from src.optopus.trades.option_leg import OptionLeg
from src.optopus.trades.exit_conditions import (
    DefaultExitCondition,
    CompositeExitCondition,
    TimeBasedCondition,
    ProfitTargetCondition,
    StopLossCondition,
)
from loguru import logger


class TestOptionStrategy(unittest.TestCase):

    def setUp(self):
        self.entry_df = pd.read_parquet("data/SPY_2024-09-06 15-30.parquet")
        self.update_df = pd.read_parquet("data/SPY_2024-09-06 15-45.parquet")
        self.update_df2 = pd.read_parquet("data/SPY_2024-09-09 09-45.parquet")

    def test_vertical_spread_creation(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike="ATM-1.5%",
            short_strike="ATM",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        for leg in vertical_spread.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side} | {leg.entry_bid} | {leg.entry_ask}")
        print(vertical_spread.entry_bid, vertical_spread.entry_ask)
        self.assertEqual(vertical_spread.entry_bid, vertical_spread.legs[1].entry_bid - vertical_spread.legs[0].entry_ask)
        self.assertEqual(len(vertical_spread.legs), 2)
        self.assertEqual(vertical_spread.strategy_type, "Vertical Spread")

    def test_iron_condor_creation(self):
        iron_condor = IronCondor.create_iron_condor(
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
        for leg in iron_condor.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side}")
        self.assertEqual(len(iron_condor.legs), 4)
        self.assertEqual(iron_condor.strategy_type, "Iron Condor")

    def test_straddle_creation(self):
        straddle = Straddle.create_straddle(
            symbol="SPY",
            strike="ATM+1",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        for leg in straddle.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side}")
        self.assertEqual(straddle.legs[0].strike, straddle.legs[1].strike)
        self.assertEqual(len(straddle.legs), 2)
        self.assertEqual(straddle.strategy_type, "Straddle")

    def test_butterfly_creation(self):
        butterfly = IronButterfly.create_iron_butterfly(
            symbol="SPY",
            lower_strike=540,
            middle_strike=550,
            upper_strike=560,
            expiration="2024-10-31",
            strategy_side="CREDIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        for leg in butterfly.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side}")
        self.assertEqual(butterfly.legs[1].strike, butterfly.legs[2].strike)
        self.assertEqual(len(butterfly.legs), 4)
        self.assertEqual(butterfly.strategy_type, "Iron Butterfly")

    def test_naked_call_creation(self):
        naked_call = NakedCall.create_naked_call(
            symbol="SPY",
            strike=550,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        for leg in naked_call.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side}")
        self.assertEqual(len(naked_call.legs), 1)
        self.assertEqual(naked_call.strategy_type, "Naked Call")

    def test_naked_put_creation(self):
        naked_put = NakedPut.create_naked_put(
            symbol="SPY",
            strike=540,
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        for leg in naked_put.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side}")
        self.assertEqual(len(naked_put.legs), 1)
        self.assertEqual(naked_put.strategy_type, "Naked Put")

    def test_strategy_update(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        # print(vertical_spread.strategy_side)
        vertical_spread.update("2024-09-06 15:30:00", self.entry_df)
        self.assertTrue(
            vertical_spread.total_pl()
            == 0 - vertical_spread.calculate_total_commission()
        )
        self.assertEqual(
            vertical_spread.current_time, pd.Timestamp("2024-09-06 15:30:00")
        )
        vertical_spread.update("2024-09-06 15:45:00", self.update_df)
        # print(vertical_spread.total_pl())
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
        spread1 = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = VerticalSpread.create_vertical_spread(
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
        spread1 = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = VerticalSpread.create_vertical_spread(
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
        spread1 = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=561,
            short_strike=551,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = VerticalSpread.create_vertical_spread(
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
        spread1 = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = VerticalSpread.create_vertical_spread(
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
        spread1 = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-11-15",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        spread2 = VerticalSpread.create_vertical_spread(
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

    def test_required_capital_credit_call_spread(self):
        credit_call_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=560,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        credit_call_capital = credit_call_spread.get_required_capital()
        expected_credit_call_capital = (
            (560 - 550 - credit_call_spread.entry_net_premium)
            * 100
            * credit_call_spread.contracts
        ) + credit_call_spread.calculate_total_commission()
        self.assertEqual(credit_call_capital, expected_credit_call_capital)

    def test_required_capital_debit_call_spread(self):
        debit_call_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike=550,
            short_strike=560,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        debit_call_capital = debit_call_spread.get_required_capital()
        expected_debit_call_capital = (
            debit_call_spread.entry_net_premium * 100 * debit_call_spread.contracts
        ) + debit_call_spread.calculate_total_commission()
        self.assertEqual(debit_call_capital, expected_debit_call_capital)

    def test_required_capital_debit_put_spread(self):
        debit_put_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike=550,
            short_strike=540,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        debit_put_capital = debit_put_spread.get_required_capital()
        expected_debit_put_capital = (
            debit_put_spread.entry_net_premium * 100 * debit_put_spread.contracts
        ) + debit_put_spread.calculate_total_commission()
        self.assertEqual(debit_put_capital, expected_debit_put_capital)

    def test_required_capital_credit_put_spread(self):
        credit_put_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike=540,
            short_strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        credit_put_capital = credit_put_spread.get_required_capital()
        expected_credit_put_capital = (
            (550 - 540 - credit_put_spread.entry_net_premium)
            * 100
            * credit_put_spread.contracts
        ) + credit_put_spread.calculate_total_commission()
        self.assertEqual(credit_put_capital, expected_credit_put_capital)

    def test_required_capital_debit_straddle(self):
        debit_straddle = Straddle.create_straddle(
            symbol="SPY",
            strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            strategy_side="DEBIT",
        )
        debit_straddle_capital = debit_straddle.get_required_capital()
        expected_debit_straddle_capital = (
            debit_straddle.entry_net_premium * 100 * debit_straddle.contracts
        ) + debit_straddle.calculate_total_commission()
        self.assertEqual(debit_straddle_capital, expected_debit_straddle_capital)

    def test_required_capital_debit_butterfly(self):
        debit_butterfly = IronButterfly.create_iron_butterfly(
            symbol="SPY",
            lower_strike=540,
            middle_strike=550,
            upper_strike=560,
            expiration="2024-12-20",
            strategy_side="DEBIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        debit_butterfly_capital = debit_butterfly.get_required_capital()
        expected_debit_butterfly_capital = (
            debit_butterfly.entry_net_premium * 100 * debit_butterfly.contracts
        ) + debit_butterfly.calculate_total_commission()
        self.assertEqual(debit_butterfly_capital, expected_debit_butterfly_capital)

    def test_required_capital_debit_iron_condor(self):
        debit_iron_condor = IronCondor.create_iron_condor(
            symbol="SPY",
            put_long_strike=550,
            put_short_strike=540,
            call_short_strike=580,
            call_long_strike=570,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        debit_iron_condor_capital = debit_iron_condor.get_required_capital()
        expected_debit_iron_condor_capital = (
            debit_iron_condor.entry_net_premium * 100 * debit_iron_condor.contracts
        ) + debit_iron_condor.calculate_total_commission()
        self.assertEqual(
            debit_iron_condor_capital, expected_debit_iron_condor_capital
        )

    def test_required_capital_iron_condor(self):
        iron_condor = IronCondor.create_iron_condor(
            symbol="SPY",
            put_long_strike=540,
            put_short_strike=550,
            call_short_strike=570,
            call_long_strike=580,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        iron_condor_capital = iron_condor.get_required_capital()
        expected_iron_condor_capital = (
            (max(abs(540 - 550), abs(570 - 580)) - iron_condor.entry_net_premium)
            * 100
            * iron_condor.contracts
        ) + iron_condor.calculate_total_commission()
        self.assertEqual(iron_condor_capital, expected_iron_condor_capital)

    def test_required_capital_straddle(self):
        straddle = Straddle.create_straddle(
            symbol="SPY",
            strike=550,
            expiration="2024-12-20",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        straddle_capital = straddle.get_required_capital()
        expected_straddle_capital = (
            straddle.entry_net_premium * 100 * straddle.contracts
        ) + straddle.calculate_total_commission()
        self.assertEqual(straddle_capital, expected_straddle_capital)

    def test_required_capital_butterfly(self):
        butterfly = IronButterfly.create_iron_butterfly(
            symbol="SPY",
            lower_strike=540,
            middle_strike=550,
            upper_strike=560,
            expiration="2024-12-20",
            strategy_side="CREDIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        butterfly_capital = butterfly.get_required_capital()
        expected_butterfly_capital = (
            (butterfly.legs[1].strike - butterfly.legs[0].strike)
            * 100
            * butterfly.contracts
        ) + butterfly.calculate_total_commission()
        expected_butterfly_capital = (
            (max(abs(540 - 550), abs(550 - 560)) - butterfly.entry_net_premium)
            * 100
            * butterfly.contracts
        ) + butterfly.calculate_total_commission()
        self.assertEqual(butterfly_capital, expected_butterfly_capital)

    def test_required_capital_naked_call(self):
        naked_call = NakedCall.create_naked_call(
            symbol="SPY",
            strike=550,
            expiration="2024-12-20",
            strategy_side="DEBIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        naked_call_capital = naked_call.get_required_capital()
        expected_naked_call_capital = (
            abs(naked_call.entry_net_premium * 100 * naked_call.contracts)
        ) + naked_call.calculate_total_commission()
        self.assertEqual(naked_call_capital, expected_naked_call_capital)

    def test_required_capital_naked_put(self):
        naked_put = NakedPut.create_naked_put(
            symbol="SPY",
            strike=540,
            expiration="2024-12-20",
            strategy_side="DEBIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        naked_put_capital = naked_put.get_required_capital()
        expected_naked_put_capital = (
            abs(naked_put.entry_net_premium * 100 * naked_put.contracts)
        ) + naked_put.calculate_total_commission()
        self.assertEqual(naked_put_capital, expected_naked_put_capital)
    
    def test_required_capital_credit_naked_call(self):
        naked_call = NakedCall.create_naked_call(
            symbol="SPY",
            strike=550,
            expiration="2024-12-20",
            strategy_side="CREDIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        naked_call_capital = naked_call.get_required_capital()
        expected_naked_call_capital = (
            abs((naked_call.legs[0].strike - naked_call.entry_net_premium) * 100 * naked_call.contracts)
        ) + naked_call.calculate_total_commission()
        self.assertEqual(naked_call_capital, expected_naked_call_capital)

    def test_required_capital_credit_naked_put(self):
        naked_put = NakedPut.create_naked_put(
            symbol="SPY",
            strike=540,
            expiration="2024-12-20",
            strategy_side="CREDIT",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        naked_put_capital = naked_put.get_required_capital()
        expected_naked_put_capital = (
            abs((naked_put.legs[0].strike - naked_put.entry_net_premium) * 100 * naked_put.contracts)
        ) + naked_put.calculate_total_commission()
        self.assertEqual(naked_put_capital, expected_naked_put_capital)

    def test_dit_calculation(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
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
    
    def test_dte_calculation1(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-09-06",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(vertical_spread.entry_dte, 0)
    
    def test_dte_calculation2(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.3",
            expiration="2024-10-31",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
        )
        self.assertEqual(vertical_spread.entry_dte, 55)

    def test_close_strategy(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
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
    
    def test_close_strategy_dte(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
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
        self.assertEqual(vertical_spread.exit_dte, 55)
    
    def test_close_strategy_dit(self):
        vertical_spread = VerticalSpread.create_vertical_spread(
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
        self.assertEqual(vertical_spread.exit_dit, 0)

    def test_won_attribute(self):
        profit_target_condition = ProfitTargetCondition(profit_target=1)
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="CALL",
            long_strike="+2",
            short_strike="+0.4",
            expiration=pd.Timestamp("2024-10-31"),
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=profit_target_condition,
        )
        for leg in vertical_spread.legs:
            logger.debug(f"{leg.symbol} | {leg.expiration} | {leg.strike} | {leg.option_type} | {leg.position_side} | {leg.entry_price}")
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertFalse(vertical_spread.won)

    def test_stop_loss_condition(self):
        stop_loss_condition = StopLossCondition(stop_loss=10)
        vertical_spread = VerticalSpread.create_vertical_spread(
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

    def test_profit_target_condition(self):
        profit_target_condition = ProfitTargetCondition(profit_target=5)
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike="-2",
            short_strike="-0.3",
            expiration=5,
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=profit_target_condition,
        )
        for legs in vertical_spread.legs:
            logger.debug(f"{legs.symbol} | {legs.expiration} | {legs.strike} | {legs.option_type} | {legs.position_side} | {legs.entry_price}")
        logger.debug(vertical_spread.entry_net_premium)
        logger.debug(vertical_spread.total_pl())
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        vertical_spread.update("2024-09-09 09:45:00", self.update_df2)
        self.assertTrue(vertical_spread.won)

    def test_time_based_condition(self):
        time_based_condition = TimeBasedCondition(exit_time_before_expiration=pd.Timedelta(minutes=15))
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike="-2",
            short_strike="-0.3",
            expiration="2024-09-06",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=time_based_condition,
        )
        vertical_spread.update("2024-09-06 15:45:00", self.update_df)
        self.assertTrue(vertical_spread.status == "CLOSED")
    
    def test_default_exit_condition(self):
        default_exit_condition = DefaultExitCondition()
        vertical_spread = VerticalSpread.create_vertical_spread(
            symbol="SPY",
            option_type="PUT",
            long_strike="-2",
            short_strike="-0.3",
            expiration="2024-09-06",
            contracts=1,
            entry_time="2024-09-06 15:30:00",
            option_chain_df=self.entry_df,
            exit_scheme=default_exit_condition,
        )
        vertical_spread.update("2024-09-06 15:45:00", self.update_df)
        logger.debug(vertical_spread.return_percentage())
        self.assertTrue(vertical_spread.status == "CLOSED")


if __name__ == "__main__":
    unittest.main()
