import pandas as pd
import os
from ..trades.option_manager import OptionBacktester
from ..trades.option_spread import OptionStrategy
import numpy as np
from loguru import logger
from datetime import datetime


class Backtest:
    def __init__(
        self,
        config,
        entry_signal_file,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        strategy_params,
        debug=False,
    ):
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

    def run_backtest(self, start_date=None, end_date=None, skip_fridays=False, plot_performance=True):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        # Convert start_date and end_date to strings if they are datetime objects
        if isinstance(start_date, (pd.Timestamp, datetime)):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, (pd.Timestamp, datetime)):
            end_date = end_date.strftime('%Y-%m-%d')
        # Generate time range for trading hours
        time_range = pd.date_range(
            start=f"{start_date} {self.trading_start_time}",
            end=f"{end_date} {self.trading_end_time}",
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
        inp["isEntry"] = False
        inp.loc[inp.query(self.strategy_params["condition"]).index, "isEntry"] = True
        inp["isEntry"] = inp["isEntry"].astype(bool)
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
                    logger.warning(
                        f"No data available for {time}. Skipping this update."
                    )
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
                entry_signal = inp.loc[time, "isEntry"]
            except Exception as e:
                if self.debug:
                    logger.error(f"Error getting entry signal: {e} at {time}")
                continue

            # Create spread
            try:
                if entry_signal:
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
                            logger.info(
                                f"{time} Spread not added due to NaN required capital."
                            )
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

    def create_time_ranges(
        self, start_date: str, end_date: str, n_splits: int, years_per_split: float
    ) -> List[Tuple[str, str]]:
        """Create time ranges for cross-validation."""
        full_range = pd.date_range(start=start_date, end=end_date, freq="15min")

        full_range = pd.Series(full_range).loc[
            full_range.indexer_between_time("09:45", "15:45")
        ]
        full_range = full_range[full_range.dt.weekday < 5]
        full_range = full_range.sort_values(ascending=True)
        full_range = full_range.reset_index(drop=True)

        total_days = len(full_range)
        split_size = int(years_per_split * 252 * 26)

        ts_folds = []
        for i in range(n_splits):
            start_idx = int(i * (total_days - split_size) / (n_splits - 1))
            end_idx = start_idx + split_size
            if end_idx > total_days:
                end_idx = total_days
            fold = full_range.iloc[start_idx:end_idx]
            ts_folds.append(
                (
                    fold.iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                    fold.iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        return ts_folds
