import numpy as np


class HampelFilterNumpy:
    """Scikit-learn style Hampel filter wrapper for the NumPy implementation."""

    def __init__(self, window_size=10, n_sigma=3, k=1.4826, max_iterations=5, replace_with_na=False):
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.k = k
        self.max_iterations = max_iterations
        self.replace_with_na = replace_with_na

    def fit(self, X, y=None):
        """No fitting required, returns self for compatibility."""
        return self

    def transform(self, X):
        """Apply Hampel filter to input data"""
        half_win_length = self.window_size // 2

        y = np.array(X.copy())
        prev_bad_count = 0

        for _ in range(self.max_iterations):
            padded_y = np.concatenate(
                [[np.nan] * half_win_length, y, [np.nan] * half_win_length]
            )
            windows_y = np.ma.array(
                np.lib.stride_tricks.sliding_window_view(
                    padded_y, 2 * half_win_length + 1
                )
            )
            windows_y[np.isnan(windows_y)] = np.ma.masked
            median_y = np.ma.median(windows_y, axis=1)
            mad_y = np.ma.median(np.abs(windows_y - np.atleast_2d(median_y).T), axis=1)

            new_bad = np.abs(y - median_y) > (mad_y * self.n_sigma * self.k)
            current_bad_count = np.sum(new_bad)

            if current_bad_count == prev_bad_count:
                break

            for i in sorted(np.where(new_bad)[0]):
                if i > 0:
                    y[i] = np.nan if self.replace_with_na else y[i - 1]

            prev_bad_count = current_bad_count

        return y.reshape(-1, 1)

    def fit_transform(self, X):
        """Apply Hampel filter to input data"""
        half_win_length = self.window_size // 2

        y = np.array(X.copy())
        prev_bad_count = 0

        for _ in range(self.max_iterations):
            padded_y = np.concatenate(
                [[np.nan] * half_win_length, y, [np.nan] * half_win_length]
            )
            windows_y = np.ma.array(
                np.lib.stride_tricks.sliding_window_view(
                    padded_y, 2 * half_win_length + 1
                )
            )
            windows_y[np.isnan(windows_y)] = np.ma.masked
            median_y = np.ma.median(windows_y, axis=1)
            mad_y = np.ma.median(np.abs(windows_y - np.atleast_2d(median_y).T), axis=1)

            new_bad = np.abs(y - median_y) > (mad_y * self.n_sigma * self.k)
            current_bad_count = np.sum(new_bad)

            if current_bad_count == prev_bad_count:
                break

            for i in sorted(np.where(new_bad)[0]):
                if i > 0:
                    y[i] = np.nan if self.replace_with_na else y[i - 1]

            prev_bad_count = current_bad_count

        return y.reshape(-1, 1)
