import pandas as pd
import os
from src.optopus.trades.option_manager import OptionBacktester, Config
from src.optopus.trades.option_spread import OptionStrategy
from datetime import timedelta
import logging
import cProfile
import pstats
from pstats import SortKey
from traceback import print_tb
import numpy as np
from loguru import logger


# Configuration
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
)
SYMBOL = "SPY"
ENTRY_SIGNAL_FILE = f"~/Downloads/stockdata/{SYMBOL}-AggEMARSICCI.csv"
START_DATE = "2016-01-04"
END_DATE = "2024-10-04"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"
DEBUG = False

# Strategy parameters for vertical spreads
PUT_SPREAD_STRATEGY_PARAMS = {
    "option_type": "PUT",
    "dte": 60,
    "short_delta": "ATM",
    "long_delta": "-1",
    "profit_target": 40,
    "stop_loss": None,
    "contracts": 1000,
    "condition": "close > 0",  # and 30 < RSI < 70
    "commission": 0.5,
}

BACKTESTER_CONFIG = Config(
    initial_capital=10000,
    max_positions=10,
    max_positions_per_day=1,
    max_positions_per_week=None,
    position_size=0.1,
    ror_threshold=0,
    gain_reinvesting=False,
    verbose=False,
)


def run_backtest(
    backtester=None,
    start_date=START_DATE,
    end_date=END_DATE,
    skip_fridays=False,
    plot_performance=True,
):
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
    entry_data = ENTRY_SIGNAL_FILE
    inp = pd.read_csv(entry_data)
    inp["date"] = pd.DatetimeIndex(inp["date"])
    inp["isSPSEntry"] = False
    inp.loc[inp.query(PUT_SPREAD_STRATEGY_PARAMS["condition"]).index, "isSPSEntry"] = (
        True
    )
    inp.set_index("date", inplace=True)
    inp = inp[["isSPSEntry"]]

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
            if DEBUG:
                logger.warning(f"No data available for {time}. Skipping this update.")
            continue

        option_chain_df = pd.read_parquet(file_path)
        if not option_chain_df.empty:
            backtester.update(time, option_chain_df)
        else:
            if DEBUG:
                logger.warning(f"Data is empty for {time}. Skipping this update.")
            continue

        if skip_fridays:
            if time.weekday() == 4:
                continue

        try:
            sps_entry_signal = inp.loc[time, "isSPSEntry"]
        except Exception as e:
            if DEBUG:
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
        except Exception as e:
            if DEBUG:
                logger.error(f"Error creating new spread: {e} at {time}")
            continue

        try:
            if new_spread is not None:
                if not np.isnan(new_spread.get_required_capital()):
                    if backtester.add_spread(new_spread):
                        if DEBUG:
                            logger.info(f"  Added new spread at {time}")
                        pass
                else:
                    if DEBUG:
                        logger.info(f"{time} Spread not added due to NaN required capital.")
                    pass
        except Exception as e:
            if DEBUG:
                logger.error(f"Error adding new spread: {e} at {time}")
            continue

        if (
            prev_active_positions != len(backtester.active_trades)
            or prev_capital != backtester.capital
        ):
            if DEBUG:
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
    backtester.print_performance_summary()
    return backtester


if __name__ == "__main__":
    bt = run_backtest()
    closed_trades_df = bt.get_closed_trades_df()
    closed_trades_df["contracts"].hist()
    # closed_trades_df
