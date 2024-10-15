import abc
from src.optopus.trades.option_spread import OptionStrategy
import pandas as pd


class Order(abc.ABC, OptionStrategy):
    order_id = None
    order_status = None
    option_strategy = None
    broker = None
    symbol = None
    strategy_type = None
    contracts = None
    entry_time = None
    entry_net_premium = None
    current_time = None
    exit_time = None
    order_status = None
    status = None
    profit_target = None
    current_bid = None
    current_ask = None
    net_premium = None
    DIT = None
    
    def set_strategy(self, option_strategy: OptionStrategy):
        if self.option_strategy is None:
            self.__dict__.update(option_strategy.__dict__)
            self.option_strategy = option_strategy

    @abc.abstractproperty
    def broker(self):
        return self._broker

    @abc.abstractmethod
    def place_order(self, account_number_hash_value, payload):
        pass

    @abc.abstractmethod
    def update_order(self, new_option_chain_df: pd.DataFrame = None):
        pass

    @abc.abstractmethod
    def submit_entry(self):
        pass

    @abc.abstractmethod
    def submit_exit(self):
        pass

    @abc.abstractmethod
    def generate_entry_payload(self):
        pass

    @abc.abstractmethod
    def generate_exit_payload(self):
        pass

    @abc.abstractmethod
    def update_order_status(self):
        pass

    @abc.abstractmethod
    def cancel(self):
        pass

    @abc.abstractmethod
    def modify(self, new_payload):
        pass
    
    @abc.abstractmethod
    def market_isOpen(self):
        pass

    def __repr__(self):
        return (
            f"Order(\n"
            f"  Order ID: {self.order_id},\n"
            f"  Order Status: {self.order_status}\n"
            f")"
        )
