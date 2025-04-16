import os
import pandas as pd
from pathlib import Path


class DataProcessor:
    def __init__(self, ohlc, ticker=None):
        self.ticker = ticker
        self.ohlc = ohlc

        if isinstance(self.ohlc, str):
            # Check if it's a file path
            if Path(self.ohlc).exists():
                self.intraday_data = pd.read_csv(self.ohlc, parse_dates=["date"])
                self.intraday_data["day"] = pd.DatetimeIndex(
                    self.intraday_data.date.dt.date
                )
                self.intraday_data.set_index("date", inplace=True)

            else:
                # Check if it's a brokerage name
                if self.ohlc.lower() == "schwab":
                    if not self.ticker:
                        raise ValueError("Ticker symbol is required for Schwab data")

                else:
                    raise ValueError(f"Unsupported brokerage: {self.ohlc}")
        else:
            raise FileNotFoundError(f"OHLC data file not found: {self.ohlc}")

    def prepare_historical_data(self, time, current_price):
        """Prepare historical data with monthly resampling"""

        if isinstance(self.ohlc, str) and self.ohlc.lower() == "schwab":
            import optopus.brokers.schwab.schwab_data as sch

            self.schwab_data = sch.SchwabData(
                client_id=os.getenv("SCHWAB_CLIENT_ID"),
                client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
                redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
                token_file=os.getenv("SCHWAB_TOKEN_FILE"),
            )
            self.schwab_data.refresh_token()

            # Get historical data
            equity_price = self.schwab_data.get_price_history(
                self.ticker, "year", 3, frequency_type="daily", frequency=1
            )
            current_quote = self.schwab_data.get_quote(self.ticker)

            # Create OHLC DataFrame
            current_daily_data = pd.concat(
                [
                    equity_price,
                    pd.DataFrame(
                        {
                            "close": [current_quote["LAST_PRICE"].iloc[-1]],
                            "open": [current_quote["OPEN_PRICE"].iloc[-1]],
                            "high": [current_quote["HIGH_PRICE"].iloc[-1]],
                            "low": [current_quote["LOW_PRICE"].iloc[-1]],
                            "volume": [current_quote["TOTAL_VOLUME"].iloc[-1]],
                            "datetime": [pd.Timestamp.now().date()],
                        }
                    ),
                ],
                axis=0,
                ignore_index=True,
            ).set_index("datetime")
            current_daily_data.index.name = "date"
            current_daily_data.index = pd.DatetimeIndex(current_daily_data.index)
            current_daily_data.reset_index(inplace=True)
            current_daily_data.rename(columns={"date": "day"}, inplace=True)
            historical_data = current_daily_data.set_index("day").iloc[-500:]
            monthly_data = historical_data.resample("ME").apply(
                {
                    "close": "last",
                    "open": "first",
                    "low": "min",
                    "high": "max",
                    "volume": "sum",
                }
            ).dropna()

        else:
            current_intraday_data = self.intraday_data[:pd.Timestamp(time)-pd.Timedelta(microseconds=1)] # if data bar is labeled on left: 9:30 => 9:45 is labeled 9:30, for example. Small timedelta applied to make it exclusive
            historical_data = current_intraday_data.resample("B").apply(
                {
                    "close": "last",
                    "open": "first",
                    "low": "min",
                    "high": "max",
                    "volume": "sum",
                }
            ).dropna()[-500:]
            monthly_data = historical_data.resample("ME").apply(
                {
                    "close": "last",
                    "open": "first",
                    "low": "min",
                    "high": "max",
                    "volume": "sum",
                }
            ).dropna()

        return historical_data, monthly_data
