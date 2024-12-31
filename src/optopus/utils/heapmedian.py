import time
import pandas as pd


class ContinuousMedian:
    def __init__(self):
        self.window = []

    def add(self, num):
        if pd.isna(num):
            return
        self.window.append(num)

    def get_median(self):
        if not self.window:
            return None
        sorted_window = sorted(self.window)
        mid = len(sorted_window) // 2
        if len(sorted_window) % 2 == 0:
            return (sorted_window[mid - 1] + sorted_window[mid]) / 2
        return sorted_window[mid]

    def remove(self, num):
        if pd.isna(num):
            return
        self.window.remove(num)


def test_rolling_performance(window_size=7):
    import numpy as np

    data = (np.random.randint(-10000, 10000, 1000000) / 1231).astype(float)
    # Introduce some NaN values
    data[np.random.choice(data.size, size=int(data.size * 0.01), replace=False)] = (
        np.nan
    )

    # Test ContinuousMedian
    start_time = time.time()
    cm = ContinuousMedian()
    rolling_medians_cm = []
    window = []
    for i, num in enumerate(data):
        window.append(num)
        cm.add(num)
        if i >= window_size:
            cm.remove(window.pop(0))
        if i >= window_size - 1:
            if not any(pd.isna(window)):
                rolling_medians_cm.append(cm.get_median())
    end_time = time.time()
    print(f"ContinuousMedian: Time = {end_time - start_time:.2f} seconds")

    # Test numpy
    start_time = time.time()
    rolling_medians_np = []
    window = []
    for i, num in enumerate(data):
        window.append(num)
        if i >= window_size:
            window.pop(0)
        if i >= window_size - 1:
            if not any(pd.isna(window)):
                rolling_medians_np.append(np.median(window))
    end_time = time.time()
    print(f"NumPy: Time = {end_time - start_time:.2f} seconds")

    # Test sorting method
    start_time = time.time()
    rolling_medians_sort = []
    window = []
    for i, num in enumerate(data):
        window.append(num)
        if i >= window_size:
            window.pop(0)
        if i >= window_size - 1:
            if not any(pd.isna(window)):
                sorted_window = sorted(window)
                mid = len(sorted_window) // 2
                if len(sorted_window) % 2 == 0:
                    rolling_medians_sort.append(
                        (sorted_window[mid - 1] + sorted_window[mid]) / 2
                    )
                else:
                    rolling_medians_sort.append(sorted_window[mid])
    end_time = time.time()
    print(f"Sorting: Time = {end_time - start_time:.2f} seconds")

    # Assert that the results are the same
    assert np.allclose(
        rolling_medians_cm, rolling_medians_np
    ), "ContinuousMedian and NumPy results do not match"
    assert np.allclose(
        rolling_medians_cm, rolling_medians_sort
    ), "ContinuousMedian and Sorting results do not match"


if __name__ == "__main__":
    test_rolling_performance()
