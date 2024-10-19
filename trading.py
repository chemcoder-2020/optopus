from abc import ABC, abstractmethod

class TradingAPI(ABC):
    @abstractmethod
    def place_order(self, payload):
        pass

    @abstractmethod
    def modify_order(self, order_id, payload):
        pass
