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
        # Filter expirations to only include those that are at least target_date days from t0
        valid_expirations = [exp for exp in expirations if exp >= target_date]
        if not valid_expirations:
            raise ValueError("No valid expiration dates found that meet the target date criteria.")
        closest_expiration = min(valid_expirations, key=lambda x: abs(x - target_date))
        return closest_expiration

    def get_desired_strike(self, expiration: pd.Timestamp, option_type: str, target: float, by: str = 'delta') -> float:
        """
        Get the desired strike price at the specified expiration based on option type and target value.

        :param expiration: Expiration date as pd.Timestamp.
        :param option_type: Type of option ('CALL' or 'PUT').
        :param target: Target value (either delta or strike price).
        :param by: Method to find strike ('delta' or 'strike', defaults to 'delta').
        :return: Closest strike price available at the specified expiration.
        :raises ValueError: If invalid option_type or method is provided.
        """
        if option_type not in ['CALL', 'PUT']:
            raise ValueError("option_type must be either 'CALL' or 'PUT'")
        
        if by not in ['delta', 'strike']:
            raise ValueError("by must be either 'delta' or 'strike'")

        # Filter by expiration
        expiration_data = self.option_chain_df[
            self.option_chain_df['EXPIRE_DATE'] == expiration
        ]

        if expiration_data.empty:
            raise ValueError(f"No data found for expiration {expiration}")

        if by == 'delta':
            delta_col = 'P_DELTA' if option_type == 'PUT' else 'C_DELTA'
            target = -abs(target) if option_type == 'PUT' else target
            # Find strike with closest delta
            closest_strike = expiration_data.iloc[
                (expiration_data[delta_col] - target).abs().idxmin()
            ]['STRIKE']
        else:  # by strike
            closest_strike = min(expiration_data['STRIKE'], key=lambda x: abs(x - target))

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

    target_date_str = "2024-10-06"  # Specific date
    closest_expiration_str = converter.get_closest_expiration(target_date_str)
    print(f"Closest expiration (str): {closest_expiration_str}")

    target_date_timestamp = pd.Timestamp("2024-10-06")  # Specific date as pd.Timestamp
    closest_expiration_timestamp = converter.get_closest_expiration(target_date_timestamp)
    print(f"Closest expiration (pd.Timestamp): {closest_expiration_timestamp}")

    target_date_datetime = datetime(2024, 10, 6)  # Specific date as datetime
    closest_expiration_datetime = converter.get_closest_expiration(target_date_datetime)
    print(f"Closest expiration (datetime): {closest_expiration_datetime}")

    # Example usage of get_desired_strike
    # By delta
    target_delta = 0.30  # Desired delta
    call_strike = converter.get_desired_strike(closest_expiration_int, 'CALL', target_delta, by='delta')
    put_strike = converter.get_desired_strike(closest_expiration_int, 'PUT', target_delta, by='delta')
    print(f"CALL strike at delta {target_delta}: {call_strike}")
    print(f"PUT strike at delta {target_delta}: {put_strike}")

    # By strike
    desired_strike = 450.0  # Desired strike price
    call_strike = converter.get_desired_strike(closest_expiration_int, 'CALL', desired_strike, by='strike')
    put_strike = converter.get_desired_strike(closest_expiration_int, 'PUT', desired_strike, by='strike')
    print(f"Closest CALL strike to {desired_strike}: {call_strike}")
    print(f"Closest PUT strike to {desired_strike}: {put_strike}")
