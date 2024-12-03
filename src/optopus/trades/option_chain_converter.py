import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

class OptionChainConverter:
    def __init__(self, option_chain_df: pd.DataFrame):
        """
        Initialize the OptionChainConverter with an option chain DataFrame.

        :param option_chain_df: DataFrame containing the option chain data.
        """
        self.option_chain_df = option_chain_df
        self.option_chain_df['QUOTE_READTIME'] = self._convert_to_eastern_tz_naive(self.option_chain_df['QUOTE_READTIME'])
        self.option_chain_df['EXPIRE_DATE'] = self._convert_to_eastern_tz_naive(self.option_chain_df['EXPIRE_DATE'])

    def _convert_to_eastern_tz_naive(self, dt_series: pd.Series) -> pd.Series:
        """
        Convert a series of datetime objects to Eastern time and make them tz-naive.

        :param dt_series: Series of datetime objects.
        :return: Series of tz-naive datetime objects in Eastern time.
        """
        eastern = pytz.timezone('US/Eastern')
        if dt_series.dt.tz is not None:
            return dt_series.dt.tz_convert(eastern).dt.tz_localize(None)
        else:
            return dt_series.dt.tz_localize('UTC').dt.tz_convert(eastern).dt.tz_localize(None)

    def get_closest_expiration(self, target_date: int | pd.Timestamp | str | datetime) -> pd.Timestamp:
        """
        Get the closest expiration date to the target date.

        :param target_date: Target date for expiration (int for DTE, pd.Timestamp, str, or datetime).
        :return: Closest expiration date as pd.Timestamp.
        """
        t0 = self.option_chain_df['QUOTE_READTIME'].iloc[0]

        if isinstance(target_date, int):
            target_date = t0 + timedelta(days=target_date)
        elif isinstance(target_date, str):
            target_date = pd.to_datetime(target_date, utc=True).astimezone(pytz.timezone('US/Eastern')).replace(tzinfo=None)
        elif isinstance(target_date, datetime):
            target_date = pd.Timestamp(target_date).tz_localize('UTC').astimezone(pytz.timezone('US/Eastern')).replace(tzinfo=None)
        elif isinstance(target_date, pd.Timestamp):
            if target_date.tz is None:
                target_date = target_date.tz_localize('UTC').astimezone(pytz.timezone('US/Eastern')).replace(tzinfo=None)
            else:
                target_date = target_date.astimezone(pytz.timezone('US/Eastern')).replace(tzinfo=None)

        expirations = self.option_chain_df['EXPIRE_DATE'].unique()
        closest_expiration = min(expirations, key=lambda x: abs(x - target_date))
        return closest_expiration

    def get_desired_strike(self, expiration: pd.Timestamp, strike: float) -> float:
        """
        Get the desired strike price at the specified expiration.

        :param expiration: Expiration date as pd.Timestamp.
        :param strike: Desired strike price.
        :return: Closest strike price available at the specified expiration.
        """
        expiration_data = self.option_chain_df[self.option_chain_df['EXPIRE_DATE'] == expiration]
        closest_strike = min(expiration_data['strike'], key=lambda x: abs(x - strike))
        return closest_strike

if __name__ == "__main__":
    # Load the test data
    file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-06 15-30.parquet"
    option_chain_df = pd.read_parquet(file_path)

    # Initialize the OptionChainConverter
    converter = OptionChainConverter(option_chain_df)

    # Example usage of get_closest_expiration
    target_date_int = 30  # 30 days from QUOTE_READTIME
    closest_expiration_int = converter.get_closest_expiration(target_date_int)
    print(f"Closest expiration (int): {closest_expiration_int}")

    target_date_str = "2024-10-01"  # Specific date
    closest_expiration_str = converter.get_closest_expiration(target_date_str)
    print(f"Closest expiration (str): {closest_expiration_str}")

    target_date_timestamp = pd.Timestamp("2024-10-15")  # Specific date as pd.Timestamp
    closest_expiration_timestamp = converter.get_closest_expiration(target_date_timestamp)
    print(f"Closest expiration (pd.Timestamp): {closest_expiration_timestamp}")

    target_date_datetime = datetime(2024, 11, 1)  # Specific date as datetime
    closest_expiration_datetime = converter.get_closest_expiration(target_date_datetime)
    print(f"Closest expiration (datetime): {closest_expiration_datetime}")

    # Example usage of get_desired_strike
    desired_strike = 450.0  # Desired strike price
    closest_strike = converter.get_desired_strike(closest_expiration_int, desired_strike)
    print(f"Closest strike at {closest_expiration_int}: {closest_strike}")
