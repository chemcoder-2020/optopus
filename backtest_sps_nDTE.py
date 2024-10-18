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


class Backtest:
    def __init__(self, config, entry_signal_file, data_folder, start_date, end_date, trading_start_time, trading_end_time, strategy_params, debug=False):
        self.config = config
        self.entry_signal_file = entry_signal_file
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.trading_start_time = trading_start_time
        self.trading_end_time = trading_end_time
        self.debug = debug
        self.backtester = OptionBacktester(self.config)
        self.symbol = "SPY"
        self.strategy_params = strategy_params

    def run_backtest(self, skip_fridays=False, plot_performance=True):
        # Generate time range for trading hours
        time_range = pd.date_range(
            start=f"{self.start_date} {self.trading_start_time}",
            end=f"{self.end_date} {self.trading_end_time}",
            freq="15min",
        )

        # Filter time range to only include trading hours
        trading_times = time_range.indexer_between_time(
            self.trading_start_time, self.trading_end_time
        )
        time_range = time_range[trading_times]

        # Load entry signals
        entry_data = self.entry_signal_file
        inp = pd.read_csv(entry_data)
        inp["date"] = pd.DatetimeIndex(inp["date"])
        inp["isSPSEntry"] = False
        inp.loc[inp.query(self.strategy_params["condition"]).index, "isEntry"] = True
        inp.set_index("date", inplace=True)
        inp = inp[["isEntry"]]

        prev_active_positions = None
        prev_capital = None
        prev_year = None

        for i, time in enumerate(time_range):
            new_spread = None
            option_chain_df = None
            year = time.year

            if year != prev_year:
                logger.info(f"Processing year: {year}")
                prev_year = year
            filename = f"{self.symbol}_{time.strftime('%Y-%m-%d %H-%M')}.parquet"
            file_path = os.path.join(self.data_folder, filename)

            if not os.path.exists(file_path):
                if self.debug:
                    logger.warning(f"No data available for {time}. Skipping this update.")
                continue

            option_chain_df = pd.read_parquet(file_path)
            if not option_chain_df.empty:
                self.backtester.update(time, option_chain_df)
            else:
                if self.debug:
                    logger.warning(f"Data is empty for {time}. Skipping this update.")
                continue

            if skip_fridays:
                if time.weekday() == 4:
                    continue

            try:
                sps_entry_signal = inp.loc[time, "isSPSEntry"]
            except Exception as e:
                if self.debug:
                    logger.error(f"Error getting entry signal: {e} at {time}")
                continue

            # Create SPS spread
            try:
                if sps_entry_signal:
                    new_spread = OptionStrategy.create_vertical_spread(
                        symbol=self.symbol,
                        option_type=self.strategy_params["option_type"],
                        long_strike=self.strategy_params["long_delta"],
                        short_strike=self.strategy_params["short_delta"],
                        expiration=self.strategy_params["dte"],
                        contracts=self.strategy_params["contracts"],
                        entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        option_chain_df=option_chain_df,
                        profit_target=self.strategy_params["profit_target"],
                        stop_loss=self.strategy_params["stop_loss"],
                        commission=self.strategy_params["commission"],
                    )
            except Exception as e:
                if self.debug:
                    logger.error(f"Error creating new spread: {e} at {time}")
                continue

            try:
                if new_spread is not None:
                    if not np.isnan(new_spread.get_required_capital()):
                        if self.backtester.add_spread(new_spread):
                            if self.debug:
                                logger.info(f"  Added new spread at {time}")
                    else:
                        if self.debug:
                            logger.info(f"{time} Spread not added due to NaN required capital.")
            except Exception as e:
                if self.debug:
                    logger.error(f"Error adding new spread: {e} at {time}")
                continue

            if (
                prev_active_positions != len(self.backtester.active_trades)
                or prev_capital != self.backtester.capital
            ):
                if self.debug:
                    logger.info(
                        f"  Time: {time}, Active trades: {len(self.backtester.active_trades)}, Capital: ${self.backtester.capital:.2f}, PL: ${self.backtester.get_total_pl():.2f}"
                    )
                prev_active_positions = len(self.backtester.active_trades)
                prev_capital = self.backtester.capital

        print("\nBacktest completed!")
        print(f"Final capital: ${self.backtester.capital:.2f}")
        print(f"Total P&L: ${self.backtester.get_total_pl():.2f}")
        print(f"Closed P&L: ${self.backtester.get_closed_pl():.2f}")
        print(f"Number of closed positions: {self.backtester.get_closed_positions()}")

        if plot_performance:
            self.backtester.plot_performance()
        self.backtester.print_performance_summary()
        return self.backtester


if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = (
        "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"
    )
    ENTRY_SIGNAL_FILE = f"~/Downloads/stockdata/SPY-AggEMARSICCI.csv"
    START_DATE = "2016-01-04"
    END_DATE = "2024-10-04"
    TRADING_START_TIME = "09:45"
    TRADING_END_TIME = "15:45"
    DEBUG = False

    # Strategy parameters for vertical spreads
    STRATEGY_PARAMS = {
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

    backtest = Backtest(
        config=BACKTESTER_CONFIG,
        entry_signal_file=ENTRY_SIGNAL_FILE,
        data_folder=DATA_FOLDER,
        start_date=START_DATE,
        end_date=END_DATE,
        trading_start_time=TRADING_START_TIME,
        trading_end_time=TRADING_END_TIME,
        strategy_params=STRATEGY_PARAMS,
        debug=DEBUG
    )
    bt = backtest.run_backtest()
    closed_trades_df = bt.get_closed_trades_df()
    closed_trades_df["contracts"].hist()
    # closed_trades_df
