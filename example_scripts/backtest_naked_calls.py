import pandas as pd
import os
from option_manager import OptionBacktester, BacktesterConfig
from option_spread import OptionStrategy
from datetime import timedelta
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.disabled = True

# Configuration
DATA_FOLDER = (
    "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
)
SYMBOL = "SPY"
START_DATE = "2017-01-15"
END_DATE = "2024-10-08"
TRADING_START_TIME = "09:45"
TRADING_END_TIME = "15:45"

# Strategy parameters for naked calls
STRATEGY_PARAMS = {
    "strike": "ATM",
    "dte": 20,
    "profit_target": 30,
    "stop_loss": None,
    "contracts": 1000,
    "condition": "GoldenCross",
    "commission": 0.5,
}

BACKTESTER_CONFIG = BacktesterConfig(
    initial_capital=25000,
    max_positions=10,
    max_positions_per_day=2,
    max_positions_per_week=None,
    position_size=0.05,
    ror_threshold=0.0,
    verbose=False,
)


def run_backtest(
    backtester=None, start_date=START_DATE, end_date=END_DATE, plot_performance=True
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
    entry_data = f"~/Downloads/stockdata/{SYMBOL}-AggEMARSICCI.csv"
    inp = pd.read_csv(entry_data)
    inp["date"] = pd.DatetimeIndex(inp["date"])
    inp["isNakedCallEntry"] = False
    inp.loc[inp.query(STRATEGY_PARAMS["condition"]).index, "isNakedCallEntry"] = True
    inp.set_index("date", inplace=True)
    inp = inp[["isNakedCallEntry"]]

    prev_active_positions = None
    prev_capital = None
    prev_year = None

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

        try:
            entry_signal = inp.loc[time, "isNakedCallEntry"]
        except Exception as e:
            logger.error(f"Error getting entry signal: {e} at {time}")
            continue

        try:
            if entry_signal:
                new_spread = OptionStrategy.create_naked_call(
                    symbol=SYMBOL,
                    strike=STRATEGY_PARAMS["strike"],
                    expiration=STRATEGY_PARAMS["dte"],
                    contracts=STRATEGY_PARAMS["contracts"],
                    entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    option_chain_df=option_chain_df,
                    profit_target=STRATEGY_PARAMS["profit_target"],
                    stop_loss=STRATEGY_PARAMS["stop_loss"],
                    commission=STRATEGY_PARAMS["commission"],
                )
        except Exception as e:
            logger.error(f"Error creating new naked call: {e} at {time}")
            continue

        try:
            if new_spread is not None:
                if not np.isnan(new_spread.get_required_capital()):
                    if backtester.add_spread(new_spread):
                        logger.info(f"  Added new naked call at {time}")
                else:
                    logger.info(f"{time} Spread not added due to NaN required capital.")
        except Exception as e:
            logger.error(f"Error adding new naked call: {e} at {time}")
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
    backtester.print_performance_summary()
    return backtester


if __name__ == "__main__":
    bt = run_backtest()
    closed_trades_df = bt.get_closed_trades_df()
    closed_trades_df["contracts"].hist()
