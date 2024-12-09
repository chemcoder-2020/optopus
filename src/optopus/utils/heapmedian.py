import heapq
import time


class ContinuousMedian:
    def __init__(self):
        self.max_heap = []  # Max heap for the lower half of the data
        self.min_heap = []  # Min heap for the upper half of the data

    def add(self, num):
        if pd.isna(num):
            return
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        # Balance the heaps
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def get_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]

    def remove(self, num):
        if pd.isna(num):
            return
        if num <= -self.max_heap[0]:
            self.max_heap.remove(-num)
            heapq.heapify(self.max_heap)
        else:
            self.min_heap.remove(num)
            heapq.heapify(self.min_heap)

        # Balance the heaps after removal
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))


import pandas as pd

def test_rolling_performance(window_size=7):
    import numpy as np

    data = np.random.randint(0, 10000, 1000000).astype(float)
    # Introduce some NaN values
    data[np.random.choice(data.size, size=int(data.size * 0.01), replace=False)] = np.nan

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
            rolling_medians_np.append(np.median(window))
    end_time = time.time()
    print(f"NumPy: Time = {end_time - start_time:.2f} seconds")

    # Assert that the results are the same
    assert np.allclose(rolling_medians_cm, rolling_medians_np), "Results do not match"


if __name__ == "__main__":
    test_rolling_performance()
