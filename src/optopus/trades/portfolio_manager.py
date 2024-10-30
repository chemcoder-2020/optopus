import logging
import pandas as pd
from typing import List, Dict
from .trade_manager import TradingManager

logger = logging.getLogger(__name__)


class PortfolioManager:
    def __init__(self):
        self.trading_managers: Dict[str, TradingManager] = {}

    def register_trading_manager(self, trading_manager: TradingManager):
        """Register a new TradingManager instance to be managed."""
        if trading_manager not in self.trading_managers.values():
            self.trading_managers[trading_manager.name] = trading_manager
            logger.info(
                f"Registered new TradingManager for {trading_manager.config.symbol}"
            )

    def update_trading_managers(self, option_chain_df: pd.DataFrame):
        """Update all active TradingManager instances."""
        for tm_name, trading_manager in self.trading_managers.items():
            trading_manager.auth_refresh()
            trading_manager.update_orders(option_chain_df)

    def calculate_current_indicator(self):
        """Calculate the current indicator for the portfolio."""
        total_pl = sum(
            trading_manager.get_orders_dataframe()["Total P/L"].sum()
            for trading_manager in self.active_trading_managers
        )
        return total_pl
