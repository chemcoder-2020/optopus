import pandas as pd


class DataProcessor:
    def __init__(self, ohlc):
        self.ohlc = ohlc
        if type(self.ohlc) is str:
            self.ohlc = pd.read_csv(self.ohlc, parse_dates=["date"]).set_index("date")

        self.daily_data = (
            self.ohlc.resample("D")
            .apply(
                {
                    "close": "last",
                    "open": "first",
                    "low": "min",
                    "high": "max",
                    "volume": "sum",
                }
            )
            .dropna(subset="close")
        )
        self.daily_data.reset_index(inplace=True)
        self.daily_data.rename(columns={"date": "day"}, inplace=True)
        self.ohlc = self.ohlc.reset_index()
        self.ohlc["day"] = pd.DatetimeIndex(self.ohlc.date.dt.date)

    def prepare_historical_data(self, time, current_price):
        """Prepare historical data with monthly resampling"""
        historical_data = (
            self.daily_data.set_index("day")[: time.date()].iloc[-500:].copy()
        )
        historical_data.iloc[-1, historical_data.columns.get_loc("close")] = (
            current_price
        )
        monthly_data = historical_data["close"].resample("M").last()
        historical_data = historical_data.asfreq("D").ffill()
        return historical_data, monthly_data
