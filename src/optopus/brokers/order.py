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

    def short_description(self):
        if self.strategy_type == 'Vertical Spread':
            long_leg = next((leg for leg in self.legs if leg.position_side == 'BUY'), None)
            short_leg = next((leg for leg in self.legs if leg.position_side == 'SELL'), None)
            return (
                f"{long_leg.strike if long_leg else 'N/A'}{long_leg.option_type[0]}+ | "
                f"{short_leg.strike if short_leg else 'N/A'}{short_leg.option_type[0]}- | "
                f"Expiration: {long_leg.expiration if long_leg else 'N/A'}, "
            )
        elif self.strategy_type == 'Iron Condor':
            put_long_leg = next((leg for leg in self.legs if leg.option_type == 'PUT' and leg.position_side == 'BUY'), None)
            put_short_leg = next((leg for leg in self.legs if leg.option_type == 'PUT' and leg.position_side == 'SELL'), None)
            call_long_leg = next((leg for leg in self.legs if leg.option_type == 'CALL' and leg.position_side == 'BUY'), None)
            call_short_leg = next((leg for leg in self.legs if leg.option_type == 'CALL' and leg.position_side == 'SELL'), None)
            return (
                f"{put_long_leg.strike if put_long_leg else 'N/A'}P+ | "
                f"{put_short_leg.strike if put_short_leg else 'N/A'}P- | "
                f"{call_long_leg.strike if call_long_leg else 'N/A'}C+ | "
                f"{call_short_leg.strike if call_short_leg else 'N/A'}C- | "
                f"Expiration: {put_long_leg.expiration if put_long_leg else 'N/A'}"
            )
        elif self.strategy_type == 'Straddle':
            call_leg = next((leg for leg in self.legs if leg.option_type == 'CALL'), None)
            put_leg = next((leg for leg in self.legs if leg.option_type == 'PUT'), None)
            return (
                f"{call_leg.strike if call_leg else 'N/A'}C | "
                f"{put_leg.strike if put_leg else 'N/A'}P | "
                f"Expiration: {call_leg.expiration if call_leg else 'N/A'}"
            )
        elif self.strategy_type == 'Butterfly':
            lower_leg = next((leg for leg in self.legs if leg.position_side == 'BUY' and leg.strike == min(leg.strike for leg in self.legs)), None)
            middle_leg = next((leg for leg in self.legs if leg.position_side == 'SELL' and leg.strike == sorted(leg.strike for leg in self.legs)[1]), None)
            upper_leg = next((leg for leg in self.legs if leg.position_side == 'BUY' and leg.strike == max(leg.strike for leg in self.legs)), None)
            return (
                f"Lower Strike: {lower_leg.strike if lower_leg else 'N/A'}, "
                f"Middle Strike: {middle_leg.strike if middle_leg else 'N/A'}, "
                f"Upper Strike: {upper_leg.strike if upper_leg else 'N/A'}, "
                f"Expiration: {lower_leg.expiration if lower_leg else 'N/A'}"
            )
        elif self.strategy_type in ['Naked Call', 'Naked Put']:
            leg = self.legs[0]
            return (
                f"Strike: {leg.strike}, "
                f"Expiration: {leg.expiration}, "
                f"Option Type: {leg.option_type[0]}"
            )
        else:
            return (
                f"Order ID: {self.order_id}, "
                f"Status: {self.order_status}"
            )

    def __repr__(self):
        if self.strategy_type == 'Vertical Spread':
            long_leg = next((leg for leg in self.legs if leg.position_side == 'BUY'), None)
            short_leg = next((leg for leg in self.legs if leg.position_side == 'SELL'), None)
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
        elif self.strategy_type == 'Iron Condor':
            put_long_leg = next((leg for leg in self.legs if leg.option_type == 'PUT' and leg.position_side == 'BUY'), None)
            put_short_leg = next((leg for leg in self.legs if leg.option_type == 'PUT' and leg.position_side == 'SELL'), None)
            call_long_leg = next((leg for leg in self.legs if leg.option_type == 'CALL' and leg.position_side == 'BUY'), None)
            call_short_leg = next((leg for leg in self.legs if leg.option_type == 'CALL' and leg.position_side == 'SELL'), None)
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status},\n"
                f"  Strategy Type: {self.strategy_type},\n"
                f"  Put Long Strike: {put_long_leg.strike if put_long_leg else 'N/A'},\n"
                f"  Put Short Strike: {put_short_leg.strike if put_short_leg else 'N/A'},\n"
                f"  Call Long Strike: {call_long_leg.strike if call_long_leg else 'N/A'},\n"
                f"  Call Short Strike: {call_short_leg.strike if call_short_leg else 'N/A'},\n"
                f"  Expiration: {put_long_leg.expiration if put_long_leg else 'N/A'},\n"
                f")"
            )
        elif self.strategy_type == 'Straddle':
            call_leg = next((leg for leg in self.legs if leg.option_type == 'CALL'), None)
            put_leg = next((leg for leg in self.legs if leg.option_type == 'PUT'), None)
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status},\n"
                f"  Strategy Type: {self.strategy_type},\n"
                f"  Call Strike: {call_leg.strike if call_leg else 'N/A'},\n"
                f"  Put Strike: {put_leg.strike if put_leg else 'N/A'},\n"
                f"  Expiration: {call_leg.expiration if call_leg else 'N/A'},\n"
                f")"
            )
        elif self.strategy_type == 'Butterfly':
            lower_leg = next((leg for leg in self.legs if leg.position_side == 'BUY' and leg.strike == min(leg.strike for leg in self.legs)), None)
            middle_leg = next((leg for leg in self.legs if leg.position_side == 'SELL' and leg.strike == sorted(leg.strike for leg in self.legs)[1]), None)
            upper_leg = next((leg for leg in self.legs if leg.position_side == 'BUY' and leg.strike == max(leg.strike for leg in self.legs)), None)
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status},\n"
                f"  Strategy Type: {self.strategy_type},\n"
                f"  Lower Strike: {lower_leg.strike if lower_leg else 'N/A'},\n"
                f"  Middle Strike: {middle_leg.strike if middle_leg else 'N/A'},\n"
                f"  Upper Strike: {upper_leg.strike if upper_leg else 'N/A'},\n"
                f"  Expiration: {lower_leg.expiration if lower_leg else 'N/A'},\n"
                f")"
            )
        elif self.strategy_type in ['Naked Call', 'Naked Put']:
            leg = self.legs[0]
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status},\n"
                f"  Strategy Type: {self.strategy_type},\n"
                f"  Strike: {leg.strike},\n"
                f"  Expiration: {leg.expiration},\n"
                f"  Option Type: {leg.option_type},\n"
                f")"
            )
        else:
            return (
                f"Order(\n"
                f"  Order ID: {self.order_id},\n"
                f"  Order Status: {self.order_status}\n"
                f")"
            )
