import abc
from option_spread import OptionStrategy
import pandas as pd

class AbstractOptionOrder(abc.ABC):
    def __init__(self, option_strategy: OptionStrategy, broker="Schwab"):
        self.__dict__.update(option_strategy.__dict__)
        self.order_id = None
        self.order_status = "PENDING"
        self.option_strategy = option_strategy
        self._broker = broker
    
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

    def __repr__(self):
        return (
            f"AbstractOptionOrder(\n"
            f"  Order ID: {self.order_id},\n"
            f"  Order Status: {self.order_status}\n"
            f")"
        )
