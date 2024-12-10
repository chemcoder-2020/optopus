import pandas as pd
import os
from ..trades.option_manager import OptionBacktester
from ..trades.option_spread import OptionStrategy
import numpy as np
from loguru import logger
from datetime import datetime
from typing import List, Tuple
from .base_backtest import BaseBacktest


class BacktestNakedCall(BaseBacktest):
    def __init__(
        self,
        config,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        strategy_params,
        debug=False,
    ):
        super().__init__(
            config,
            data_folder,
            start_date,
            end_date,
            trading_start_time,
            trading_end_time,
            debug=debug,
        )
        self.strategy_params = strategy_params
        self.symbol = self.strategy_params["symbol"]

    def create_spread(self, time, option_chain_df):
        """Create a naked call for the given time and option chain."""
        try:
            new_spread = OptionStrategy.create_naked_call(
                symbol=self.symbol,
                strike=self.strategy_params["strike"],
                expiration=self.strategy_params["dte"],
                contracts=self.strategy_params["contracts"],
                entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                option_chain_df=option_chain_df,
                profit_target=self.strategy_params.get("profit_target"),
                stop_loss=self.strategy_params.get("stop_loss"),
                commission=self.strategy_params.get("commission", 0),
                exit_scheme=self.strategy_params.get("exit_scheme"),
            )
            return new_spread
        except Exception as e:
            if self.debug:
                logger.error(f"Error creating new spread: {e} at {time}")
            return None
