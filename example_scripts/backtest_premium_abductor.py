import pandas as pd
import os
from option_manager import OptionBacktester, BacktesterConfig
from option_spread import OptionStrategy
from datetime import timedelta
import logging
import cProfile
import pstats
from pstats import SortKey
from traceback import print_tb
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.disabled = True

# Configuration
# DATA_FOLDER = (
#     "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
# )
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/ORATS/SPY/by_bar"
)
SYMBOL = "SPY"
START_DATE = "2022-01-03"
END_DATE = "2024-09-18"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"

# Strategy parameters for vertical spreads
PUT_SPREAD_STRATEGY_PARAMS = {
    "option_type": "PUT",
    "dte": 1,
    "short_delta": "-0.3",
    "long_delta": "-1",
    "profit_target": 40,
    "stop_loss": None,
    "contracts": 1000,
    "condition": "close > EMA10 and CCI > 0 and RSI < 70",
    "commission": 0.5,
}

CALL_SPREAD_STRATEGY_PARAMS = {
    "option_type": "CALL",
    "dte": 1,
    "short_delta": "+0.3",
    "long_delta": "+1",
    "profit_target": 40,
    "stop_loss": None,
    "contracts": 1000,
    "condition": "close < EMA10 and CCI < 0 and RSI > 30",
    "commission": 0.5,
}

BACKTESTER_CONFIG = BacktesterConfig(
    initial_capital=25000,
    max_positions=10,
    max_positions_per_day=2,
    max_positions_per_week=None,
    position_size=0.05,
    ror_threshold=0,
    verbose=False,
)


def run_backtest(backtester=None, start_date=START_DATE, end_date=END_DATE, skip_fridays=False, plot_performance=True):
    # Initialize backtester if not provided

    if backtester is None:
        backtester = OptionBacktester(BACKTESTER_CONFIG)

    # Generate time range for trading hours
    time_range = pd.date_range(
        start=f"{start_date} {TRADING_START_TIME}",
        end=f"{end_date} {TRADING_END_TIME}",
        freq="15min",
    )

    # Filter time range to only include trading hours
    trading_times = time_range.indexer_between_time(
        TRADING_START_TIME, TRADING_END_TIME
    )
    time_range = time_range[trading_times]

    # Load entry signals
    entry_data = f"~/Downloads/stockdata/{SYMBOL}-AggEMARSICCI.csv"
    inp = pd.read_csv(entry_data)
    inp["date"] = pd.DatetimeIndex(inp["date"])
    inp["isSPSEntry"] = False
    inp["isSCSEntry"] = False
    inp.loc[inp.query(PUT_SPREAD_STRATEGY_PARAMS["condition"]).index, "isSPSEntry"] = (
        True
    )
    inp.loc[inp.query(CALL_SPREAD_STRATEGY_PARAMS["condition"]).index, "isSCSEntry"] = (
        True
    )
    inp.set_index("date", inplace=True)
    inp = inp[["isSPSEntry", "isSCSEntry"]]

    prev_active_positions = None
    prev_capital = None
    prev_year = None

    # profiler = cProfile.Profile()
    # profiler.enable()

    for i, time in enumerate(time_range):
        new_spread = None
        option_chain_df = None
        year = time.year

        if year != prev_year:
            # print(year)
            prev_year = year
        filename = f"{SYMBOL}_{time.strftime('%Y-%m-%d %H-%M')}.parquet"
        file_path = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(file_path):
            logger.warning(f"No data available for {time}. Skipping this update.")
            continue

        option_chain_df = pd.read_parquet(file_path)
        if not option_chain_df.empty:
            backtester.update(time, option_chain_df)
        else:
            logger.warning(f"Data is empty for {time}. Skipping this update.")
            continue

        if skip_fridays:
            if time.weekday() == 4:
                continue

        try:
            sps_entry_signal = inp.loc[time, "isSPSEntry"]
            scs_entry_signal = inp.loc[time, "isSCSEntry"]
        except Exception as e:
            logger.error(f"Error getting entry signal: {e} at {time}")
            continue

        # Create SPS spread
        try:
            if sps_entry_signal:
                new_spread = OptionStrategy.create_vertical_spread(
                    symbol=SYMBOL,
                    option_type=PUT_SPREAD_STRATEGY_PARAMS["option_type"],
                    long_strike=PUT_SPREAD_STRATEGY_PARAMS["long_delta"],
                    short_strike=PUT_SPREAD_STRATEGY_PARAMS["short_delta"],
                    expiration=PUT_SPREAD_STRATEGY_PARAMS["dte"],
                    contracts=PUT_SPREAD_STRATEGY_PARAMS["contracts"],
                    entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    option_chain_df=option_chain_df,
                    profit_target=PUT_SPREAD_STRATEGY_PARAMS["profit_target"],
                    stop_loss=PUT_SPREAD_STRATEGY_PARAMS["stop_loss"],
                    commission=PUT_SPREAD_STRATEGY_PARAMS["commission"],
                )
            elif scs_entry_signal:
                new_spread = OptionStrategy.create_vertical_spread(
                    symbol=SYMBOL,
                    option_type=CALL_SPREAD_STRATEGY_PARAMS["option_type"],
                    long_strike=CALL_SPREAD_STRATEGY_PARAMS["long_delta"],
                    short_strike=CALL_SPREAD_STRATEGY_PARAMS["short_delta"],
                    expiration=CALL_SPREAD_STRATEGY_PARAMS["dte"],
                    contracts=CALL_SPREAD_STRATEGY_PARAMS["contracts"],
                    entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    option_chain_df=option_chain_df,
                    profit_target=CALL_SPREAD_STRATEGY_PARAMS["profit_target"],
                    stop_loss=CALL_SPREAD_STRATEGY_PARAMS["stop_loss"],
                    commission=CALL_SPREAD_STRATEGY_PARAMS["commission"],
                )
        except Exception as e:
            logger.error(f"Error creating new spread: {e} at {time}")
            # print(f"Error creating new spread: {e} at {time}")
            continue

        try:
            if new_spread is not None:
                if not np.isnan(new_spread.get_required_capital()):
                    if backtester.add_spread(new_spread):
                        logger.info(f"  Added new spread at {time}")
                else:
                    logger.info(f"{time} Spread not added due to NaN required capital.")
                    # print(new_spread)
                    # break
        except Exception as e:
            logger.error(f"Error adding new spread: {e} at {time}")
            # (f"Error adding new spread: {e} at {time}")
            continue

        if (
            prev_active_positions != len(backtester.active_trades)
            or prev_capital != backtester.capital
        ):
            logger.info(
                f"  Time: {time}, Active trades: {len(backtester.active_trades)}, Capital: ${backtester.capital:.2f}, PL: ${backtester.get_total_pl():.2f}"
            )
            prev_active_positions = len(backtester.active_trades)
            prev_capital = backtester.capital

    print("\nBacktest completed!")
    print(f"Final capital: ${backtester.capital:.2f}")
    print(f"Total P&L: ${backtester.get_total_pl():.2f}")
    print(f"Closed P&L: ${backtester.get_closed_pl():.2f}")
    print(f"Number of closed positions: {backtester.get_closed_positions()}")

    if plot_performance:
        backtester.plot_performance()
    # closed_trades_df = backtester.get_closed_trades_df()
    backtester.print_performance_summary()
    return backtester


if __name__ == "__main__":
    bt = run_backtest()
    # bt.closed_trades[-1]
    closed_trades_df = bt.get_closed_trades_df()
    closed_trades_df['contracts'].hist()
    closed_pl = pd.DataFrame(bt.performance_data).set_index("time")['closed_pl']
    # closed_pl.resample("M").last().diff().mean()
