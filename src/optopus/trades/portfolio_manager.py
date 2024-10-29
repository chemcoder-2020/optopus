import logging
import pandas as pd
from typing import List, Dict
from .trade_manager import TradingManager

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self):
        self.trading_managers: Dict[str, TradingManager] = {}
        self.active_trading_managers: List[TradingManager] = []
        self.closed_trading_managers: List[TradingManager] = []

    def register_trading_manager(self, trading_manager: TradingManager):
        """Register a new TradingManager instance to be managed."""
        if trading_manager not in self.trading_managers.values():
            self.trading_managers[trading_manager.config.symbol] = trading_manager
            self.active_trading_managers.append(trading_manager)
            logger.info(f"Registered new TradingManager for {trading_manager.config.symbol}")

    def update_trading_managers(self, option_chain_df: pd.DataFrame):
        """Update all active TradingManager instances."""
        for trading_manager in self.active_trading_managers:
            trading_manager.update_orders(option_chain_df)

    def calculate_current_indicator(self):
        """Calculate the current indicator for the portfolio."""
        total_pl = sum(trading_manager.get_orders_dataframe()["Total P/L"].sum() for trading_manager in self.active_trading_managers)
        return total_pl

    def close_trading_manager(self, symbol: str):
        """Close a TradingManager instance by its symbol."""
        if symbol in self.trading_managers:
            trading_manager = self.trading_managers[symbol]
            trading_manager.liquidate_all()
            self.closed_trading_managers.append(trading_manager)
            self.active_trading_managers.remove(trading_manager)
            del self.trading_managers[symbol]
            logger.info(f"Closed TradingManager for {symbol}")
        else:
            logger.warning(f"TradingManager for {symbol} not found.")

    def get_active_trading_managers(self) -> List[TradingManager]:
        """Get a list of active TradingManager instances."""
        return self.active_trading_managers

    def get_closed_trading_managers(self) -> List[TradingManager]:
        """Get a list of closed TradingManager instances."""
        return self.closed_trading_managers
