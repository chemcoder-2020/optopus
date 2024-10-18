import abc
from ..trades.option_spread import OptionStrategy
import pandas as pd


class Order(abc.ABC, OptionStrategy):

    def __init__(self, option_strategy: OptionStrategy):
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
    def close_order(self):
        pass
    
    @abc.abstractmethod
    def market_isOpen(self):
        pass

    def __repr__(self):
        if self.strategy_type == 'Vertical Spread':
            long_leg = next((leg for leg in self.legs if leg.option_type == 'CALL'), None)
            short_leg = next((leg for leg in self.legs if leg.option_type == 'PUT'), None)
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status},\n"
                f"  Strategy Type: {self.strategy_type},\n"
                f"  Long Strike: {long_leg.strike if long_leg else 'N/A'},\n"
                f"  Short Strike: {short_leg.strike if short_leg else 'N/A'},\n"
                f"  Expiration: {long_leg.expiration if long_leg else 'N/A'},\n"
                f"  Option Type: {long_leg.option_type if long_leg else 'N/A'}\n"
                f")"
            )
        else:
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status}\n"
                f")"
            )
