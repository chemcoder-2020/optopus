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

            else:
                # Check if it's a brokerage name
                if self.ohlc.lower() == "schwab":
                    if not self.ticker:
                        raise ValueError("Ticker symbol is required for Schwab data")
                    import optopus.brokers.schwab.schwab_data as sch
                    self.schwab_data = sch.SchwabData(
                        client_id=os.getenv("SCHWAB_CLIENT_ID"),
                        client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
                        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
                        token_file=os.getenv("SCHWAB_TOKEN_FILE"),
                    )
                    
                else:
                    raise ValueError(f"Unsupported brokerage: {self.ohlc}")
        else:
            raise FileNotFoundError(f"OHLC data file not found: {self.ohlc}")

        
    def prepare_historical_data(self, time, current_price):
        """Prepare historical data with monthly resampling"""

        if isinstance(self.ohlc, str) and self.ohlc.lower() == "schwab":
            self.schwab_data.refresh_token()
                    
            # Get historical data
            equity_price = self.schwab_data.get_price_history(
                self.ticker, "year", 3, frequency_type="daily", frequency=1
            )
            current_quote = self.schwab_data.get_quote(self.ticker)
            
            # Create OHLC DataFrame
            self.daily_data = pd.concat(
                [
                    equity_price,
                    pd.DataFrame({
                        "close": [current_quote["LAST_PRICE"].iloc[-1]],
                        "datetime": [pd.Timestamp.now().date()]
                    })
                ],
                axis=0,
                ignore_index=True
            ).set_index("datetime")
            self.daily_data.index.name = "date"
            self.daily_data.index = pd.DatetimeIndex(self.daily_data.index)
            self.daily_data.reset_index(inplace=True)
            self.daily_data.rename(columns={"date": "day"}, inplace=True)
            
        historical_data = (
            self.daily_data.set_index("day")[: time.date()].iloc[-500:].copy()
        )
        historical_data.iloc[-1, historical_data.columns.get_loc("close")] = (
            current_price
        )
        monthly_data = historical_data["close"].resample("M").last()
        historical_data = historical_data.asfreq("D").ffill()
        return historical_data, monthly_data
