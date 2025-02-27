from .base import EntryConditionChecker
from .premium_filter import PremiumFilter
from .premium_list_init import PremiumListInit
from loguru import logger
import numpy as np


class PremiumProcessCondition(EntryConditionChecker):
    def __init__(
        self, bid_mark_fluctuation=0.1, filter_method="HampelFilterNumpy", **kwargs
    ):
        self.premium_filter = PremiumFilter(filter_method=filter_method, **kwargs)
        self.premium_init = PremiumListInit()
        self.bid_mark_fluctuation = bid_mark_fluctuation
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def should_enter(self, strategy, manager, time) -> bool:
        # Init the premium context
        self.premium_init.preprocess(strategy, manager)

        # Apply the filter
        filtered_mark = self.premium_filter.preprocess(strategy, manager)

        # Check if the mark is close to the bid
        bid = strategy.current_bid
        if not np.isclose(filtered_mark, bid, rtol=self.bid_mark_fluctuation):
            logger.warning(
                f"Premium is not close to the bid. Filtered mark: {filtered_mark}, Bid: {bid}, Fluctuation: {self.bid_mark_fluctuation}"
            )
            return False

        logger.info(
            f"Passed PremiumProcessCondition check. Filtered mark: {filtered_mark}, Bid: {bid}, Fluctuation: {self.bid_mark_fluctuation}"
        )

        return True
