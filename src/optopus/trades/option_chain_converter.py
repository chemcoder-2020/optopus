import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class OptionChainConverter:
    def __init__(self, option_chain_df: pd.DataFrame):
        """
        Initialize the OptionChainConverter with an option chain DataFrame.

        :param option_chain_df: DataFrame containing the option chain data.
        """
        self.option_chain_df = option_chain_df

    def get_closest_expiration(self, target_date: int | pd.Timestamp | str | datetime) -> pd.Timestamp:
        """
        Get the closest expiration date to the target date.

        :param target_date: Target date for expiration (int for DTE, pd.Timestamp, str, or datetime).
        :return: Closest expiration date as pd.Timestamp.
        """
        if isinstance(target_date, int):
            target_date = datetime.now() + timedelta(days=target_date)
        elif isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, datetime):
            target_date = pd.to_datetime(target_date)

        expirations = self.option_chain_df['expiration'].unique()
        closest_expiration = min(expirations, key=lambda x: abs(x - target_date))
        return closest_expiration

    def get_desired_strike(self, expiration: pd.Timestamp, strike: float) -> float:
        """
        Get the desired strike price at the specified expiration.

        :param expiration: Expiration date as pd.Timestamp.
        :param strike: Desired strike price.
        :return: Closest strike price available at the specified expiration.
        """
        expiration_data = self.option_chain_df[self.option_chain_df['expiration'] == expiration]
        closest_strike = min(expiration_data['strike'], key=lambda x: abs(x - strike))
        return closest_strike
