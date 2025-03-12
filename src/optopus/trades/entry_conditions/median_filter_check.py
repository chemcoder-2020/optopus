from .base import EntryConditionChecker
from ...utils.filters import HampelFilterNumpy
from ...utils.heapmedian import ContinuousMedian
import numpy as np
from loguru import logger


class MedianCalculator(EntryConditionChecker):
    def __init__(self, window_size=7, fluctuation=0.1, method="HampelFilter", **kwargs):
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.method = method
        self.kwargs = kwargs
        if method == "ContinuousMedian":
            self.median_calculator = ContinuousMedian()
        else:
            self.median_calculator = HampelFilterNumpy(
                window_size=window_size,
                n_sigma=self.kwargs.get("n_sigma", 3),
                k=self.kwargs.get("k", 1.4826),
                max_iterations=self.kwargs.get("max_iterations", 5),
                replace_with_na=True,
            )

    def should_enter(self, strategy, manager, time) -> bool:
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2  # if bid != 0 else ask

        if not hasattr(manager, "context"):
            manager.context = {}

        if "premiums" not in manager.context:
            manager.context["premiums"] = []

        manager.context["premiums"].append(mark)
        if self.method == "ContinuousMedian":
            self.median_calculator.add(mark)

        if self.method == "ContinuousMedian":
            if len(manager.context["premiums"]) > self.window_size:
                self.median_calculator.remove(manager.context["premiums"].pop(0))
        else:
            if len(manager.context["premiums"]) > self.window_size + 1:
                manager.context["premiums"].pop(0)

        if self.method == "ContinuousMedian":
            filtered_mark = self.median_calculator.get_median()
        else:
            if len(manager.context["premiums"]) < self.window_size + 1:
                filtered_mark = 0
            else:
                filtered_mark = self.median_calculator.fit_transform(
                    np.array(manager.context["premiums"])
                ).flatten()[-1]

        if filtered_mark == 0:
            logger.warning(
                f"Filtered mark is 0. Probably not enough data for MedianCalculator's window size of {self.window_size}"
            )
            return False
        elif np.isnan(filtered_mark):
            logger.warning(
                f"Filtered mark is NaN. The filter {self.method} has detected an outlying price. Returning False and replacing premium list with previous value."
            )
            manager.context["premiums"][-1] = manager.context["premiums"][-2]
            return False

        logger.info(
            f"Passed MedianCalculator check. Filtered mark: {filtered_mark}, Bid: {bid}, Mark: {mark}, Fluctuation: {self.fluctuation}"
        )

        return np.isclose(mark, filtered_mark, rtol=self.fluctuation) and np.isclose(
            bid, mark, rtol=self.fluctuation
        )
