import pandas as pd
from datetime import datetime, timedelta
import pytz


class OptionChainConverter:
    def __init__(self, option_chain_df: pd.DataFrame):
        """
        Initialize the OptionChainConverter with an option chain DataFrame.

        :param option_chain_df: DataFrame containing the option chain data.
        """
        self.option_chain_df = option_chain_df.copy()
        self.option_chain_df["QUOTE_READTIME"] = self._convert_to_eastern_tz_naive(
            self.option_chain_df["QUOTE_READTIME"]
        )
        self.option_chain_df["EXPIRE_DATE"] = pd.to_datetime(
            self._convert_to_eastern_tz_naive(self.option_chain_df["EXPIRE_DATE"]).date
        )

    def _convert_to_eastern_tz_naive(self, dt_series: pd.Series) -> pd.Series:
        """
        Convert a series of datetime objects to Eastern time and make them tz-naive.

        :param dt_series: Series of datetime objects.
        :return: Series of tz-naive datetime objects in Eastern time.
        """
        return pd.DatetimeIndex(dt_series).tz_localize(None)

    def get_closest_expiration(
        self,
        target_date: int | pd.Timestamp | str | datetime,
        max_extra_dte: int | None = None,
    ) -> pd.Timestamp:
        """
        Get the closest expiration date to the target date.

        :param target_date: Target date for expiration (int for DTE,
                            pd.Timestamp, str, or datetime).
        :param max_extra_dte: Maximum number of extra days allowed beyond
                               target_date. If None, no upper limit.
                               Defaults to None.
        :return: Closest expiration date as pd.Timestamp.
        :raises ValueError: If no valid expiration dates are found.
        """
        t0 = self.option_chain_df["QUOTE_READTIME"].iloc[0]

        if isinstance(target_date, int):
            target_date = t0 + timedelta(days=target_date)
        elif isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, datetime):
            target_date = pd.Timestamp(target_date)
            if target_date.tz is None:
                pass
            else:
                target_date = target_date.astimezone(
                    pytz.timezone("US/Eastern")
                ).replace(tzinfo=None)
        elif isinstance(target_date, pd.Timestamp):
            if target_date.tz is None:
                pass
            else:
                target_date = target_date.astimezone(
                    pytz.timezone("US/Eastern")
                ).replace(tzinfo=None)

        target_date = pd.Timestamp(target_date.date())
        expirations = self.option_chain_df["EXPIRE_DATE"].unique()

        # Modified expiration filtering logic
        if max_extra_dte is not None:
            upper_bound = target_date + pd.Timedelta(days=max_extra_dte)
            valid_expirations = [
                exp
                for exp in expirations
                if (exp >= target_date) and (exp <= upper_bound)
            ]
        else:
            valid_expirations = [exp for exp in expirations if exp >= target_date]
        if not valid_expirations:
            raise ValueError(
                "No valid expiration dates found that meet the target date " "criteria."
            )
        closest_expiration = min(valid_expirations, key=lambda x: abs(x - target_date))
        return closest_expiration

    def get_atm_strike(
        self,
        expiration: int | pd.Timestamp | str | datetime,
        max_extra_dte: int | None = None,
    ) -> float:
        """
        Get the At-The-Money (ATM) strike price for the given expiration.

        :param expiration: Target date for expiration (int for DTE,
                           pd.Timestamp, str, or datetime).
        :param max_extra_dte: Maximum number of extra days allowed beyond
                               expiration date. If None, no upper limit.
        :return: The ATM strike price.
        :raises ValueError: If no data is found for the given expiration.
        """
        # Get the closest expiration date
        closest_expiration = self.get_closest_expiration(expiration, max_extra_dte)

        # Filter by expiration
        expiration_data = self.option_chain_df[
            self.option_chain_df["EXPIRE_DATE"].eq(closest_expiration)
        ].reset_index(drop=True)

        if expiration_data.empty:
            raise ValueError(f"No data found for expiration {expiration}")

        # Get the underlying price
        underlying_price = expiration_data["UNDERLYING_LAST"].iloc[0]

        # Find the closest strike to the underlying price
        atm_strike = min(
            expiration_data["STRIKE"].unique(), key=lambda x: abs(x - underlying_price)
        )

        return atm_strike

    def get_strike_relative_to_atm(
        self,
        expiration: int | pd.Timestamp | str | datetime,
        offset: float,
        by: str = "dollar",
        max_extra_dte: int | None = None,
    ) -> float:
        """
        Get a strike price relative to the ATM strike.

        :param expiration: Target date for expiration (int for DTE,
                           pd.Timestamp, str, or datetime).
        :param offset: Offset from ATM strike (in dollars or percentage
                       based on 'by' parameter).
        :param by: Method to calculate offset ('dollar' or 'percent',
                   defaults to 'dollar').
        :param max_extra_dte: Maximum number of extra days allowed beyond
                               expiration date. If None, no upper limit.
        :return: The strike price offset from ATM.
        :raises ValueError: If invalid offset method is provided.
        """
        if by not in ["dollar", "percent"]:
            raise ValueError("by must be either 'dollar' or 'percent'")
        atm_strike = self.get_atm_strike(expiration, max_extra_dte)

        # Get the closest expiration date
        closest_expiration = self.get_closest_expiration(expiration, max_extra_dte)

        # Filter by expiration
        expiration_data = self.option_chain_df[
            self.option_chain_df["EXPIRE_DATE"].eq(closest_expiration)
        ].reset_index(drop=True)

        # Calculate target strike based on offset method
        if by == "dollar":
            target_strike = atm_strike + offset
        else:  # by == 'percent'
            target_strike = atm_strike * (1 + offset / 100)
        relative_strike = min(
            expiration_data["STRIKE"].unique(),
            key=lambda x: abs(x - target_strike),
        )

        return relative_strike

    def get_desired_strike(
        self,
        expiration: int | pd.Timestamp | str | datetime,
        option_type: str,
        target: float,
        by: str = "delta",
        max_extra_dte: int | None = None,
    ) -> float:
        """
        Get the desired strike price at the specified expiration based on
        option type and target value.

        :param expiration: Target date for expiration (int for DTE,
                           pd.Timestamp, str, or datetime).
        :param option_type: Type of option ('CALL' or 'PUT').
        :param target: Target value (delta, strike price, or offset from ATM).
        :param by: Method to find strike ('delta', 'strike', 'atm', or
                   'atm_percent', defaults to 'delta').
        :param max_extra_dte: Maximum number of extra days allowed beyond
                               expiration date. If None, no upper limit.
        :return: Closest strike price available at the specified expiration.
        :raises ValueError: If invalid option_type or method is provided.
        """
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("option_type must be either 'CALL' or 'PUT'")

        if by not in ["delta", "strike", "atm", "atm_percent"]:
            raise ValueError(
                "by must be either 'delta', 'strike', 'atm', or " "'atm_percent'"
            )

        # Get the closest expiration date
        closest_expiration = self.get_closest_expiration(expiration, max_extra_dte)

        # Filter by expiration
        expiration_data = self.option_chain_df[
            self.option_chain_df["EXPIRE_DATE"].eq(closest_expiration)
        ].reset_index(drop=True)

        if expiration_data.empty:
            raise ValueError(f"No data found for expiration {expiration}")

        if by == "delta":
            delta_col = "P_DELTA" if option_type == "PUT" else "C_DELTA"
            target = -abs(target) if option_type == "PUT" else target
            # Find strike with closest delta
            try:
                # Calculate absolute difference
                diff_series = (expiration_data[delta_col] - target).abs()

                # Explicitly check if all values are NaN before calling idxmin()
                if diff_series.isnull().all():
                    raise ValueError(
                        f"Delta data for {option_type} at expiration "
                        f"{closest_expiration} is all NaN. Cannot find "
                        f"closest delta."
                    )

                # Find the index of the minimum difference
                idx_min = diff_series.idxmin()
                closest_strike = expiration_data.loc[idx_min]["STRIKE"]
            except ValueError as e:
                # Catch ValueError from isnull().all() check or idxmin()
                raise ValueError(
                    f"Could not find a strike for {option_type} with target "
                    f"delta {target} at expiration {closest_expiration}. "
                    f"Delta data might be missing or invalid. Original "
                    f"error: {e}"
                )
        elif by == "strike":
            target = float(target)
            # Ensure we handle potential NaNs here too
            # Find the strike closest to the target strike price
            valid_strikes = expiration_data["STRIKE"].dropna()
            if valid_strikes.empty:
                raise ValueError(
                    f"No valid strikes found for expiration " f"{closest_expiration}."
                )
            # Use idxmin on the absolute difference of valid strikes
            closest_strike = valid_strikes.loc[(valid_strikes - target).abs().idxmin()]
        elif by == "atm":
            closest_strike = self.get_strike_relative_to_atm(
                expiration,
                target,
                by="dollar",
                max_extra_dte=max_extra_dte,
            )
        else:  # by == 'atm_percent'
            closest_strike = self.get_strike_relative_to_atm(
                expiration,
                target,
                by="percent",
                max_extra_dte=max_extra_dte,
            )

        return closest_strike


if __name__ == "__main__":
    # Load the test data
    # file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-06 15-30.parquet"
    file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-09 09-45.parquet"
    # file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2019-01-24 10-30.parquet"
    option_chain_df = pd.read_parquet(file_path)

    # Initialize the OptionChainConverter
    converter = OptionChainConverter(option_chain_df)
    # converter.get_desired_strike(pd.Timestamp("2024-09-06"), "PUT", -0.05, by="delta")
    converter.get_desired_strike(pd.Timestamp("2019-01-24"), "PUT", 1, by="atm")
    converter.get_closest_expiration(0)

    # Example usage of get_closest_expiration
    target_date_int = 30  # 30 days from QUOTE_READTIME
    closest_expiration_int = converter.get_closest_expiration(
        target_date_int, max_extra_dte=5
    )
    print(f"Closest expiration (int): {closest_expiration_int}")

    target_date_str = "2024-10-06"  # Specific date
    closest_expiration_str = converter.get_closest_expiration(target_date_str)
    print(f"Closest expiration (str): {closest_expiration_str}")

    target_date_timestamp = pd.Timestamp("2024-10-06")  # Specific date as pd.Timestamp
    closest_expiration_timestamp = converter.get_closest_expiration(
        target_date_timestamp
    )
    print(f"Closest expiration (pd.Timestamp): {closest_expiration_timestamp}")

    target_date_datetime = datetime(2024, 10, 6)  # Specific date as datetime
    closest_expiration_datetime = converter.get_closest_expiration(target_date_datetime)
    print(f"Closest expiration (datetime): {closest_expiration_datetime}")

    # Example usage of get_desired_strike with different expiration types
    print("\nDemonstrating get_desired_strike with different expiration types:")

    target_delta = 0.30
    expiry_as_dte = 30  # 30 days from now

    # Using integer (DTE)
    call_strike_dte = converter.get_desired_strike(
        expiry_as_dte, "CALL", target_delta, by="delta"
    )
    put_strike_dte = converter.get_desired_strike(
        expiry_as_dte, "PUT", target_delta, by="delta"
    )
    print(f"Using DTE ({expiry_as_dte}):")
    print(f"  CALL strike at delta {target_delta}: {call_strike_dte}")
    print(f"  PUT strike at delta {target_delta}: {put_strike_dte}")

    # Using string date
    expiry_as_str = "2024-10-10"  # Same date as DTE example
    call_strike_str = converter.get_desired_strike(
        expiry_as_str, "CALL", target_delta, by="delta"
    )
    put_strike_str = converter.get_desired_strike(
        expiry_as_str, "PUT", target_delta, by="delta"
    )
    print(f"Using string date ({expiry_as_str}):")
    print(f"  CALL strike at delta {target_delta}: {call_strike_str}")
    print(f"  PUT strike at delta {target_delta}: {put_strike_str}")

    # Using pd.Timestamp
    expiry_as_timestamp = pd.Timestamp("2024-10-10")  # Same date
    call_strike_ts = converter.get_desired_strike(
        expiry_as_timestamp,
        "CALL",
        target_delta,
        by="delta",
    )
    put_strike_ts = converter.get_desired_strike(
        expiry_as_timestamp, "PUT", target_delta, by="delta"
    )
    print(f"Using pd.Timestamp ({expiry_as_timestamp}):")
    print(f"  CALL strike at delta {target_delta}: {call_strike_ts}")
    print(f"  PUT strike at delta {target_delta}: {put_strike_ts}")

    # Using datetime
    expiry_as_datetime = datetime(2024, 10, 10)  # Same date
    call_strike_dt = converter.get_desired_strike(
        expiry_as_datetime, "CALL", target_delta, by="delta"
    )
    put_strike_dt = converter.get_desired_strike(
        expiry_as_datetime, "PUT", target_delta, by="delta"
    )
    print(f"Using datetime ({expiry_as_datetime}):")
    print(f"  CALL strike at delta {target_delta}: {call_strike_dt}")
    print(f"  PUT strike at delta {target_delta}: {put_strike_dt}")

    # Example using strike price instead of delta
    print("\nDemonstrating get_desired_strike with strike price:")
    desired_strike = 450.0
    closest_call_strike = converter.get_desired_strike(
        expiry_as_dte, "CALL", desired_strike, by="strike"
    )
    closest_put_strike = converter.get_desired_strike(
        expiry_as_dte, "PUT", desired_strike, by="strike"
    )
    print(f"Closest CALL strike to {desired_strike}: {closest_call_strike}")
    print(f"Closest PUT strike to {desired_strike}: {closest_put_strike}")

    # Example using ATM and ATM relative strikes
    print("\nDemonstrating ATM strike selection:")
    atm_strike = converter.get_atm_strike(expiry_as_dte)
    print(f"ATM strike: {atm_strike}")

    # ATM+1 and ATM-1 examples
    print("\nDemonstrating ATM relative strikes:")
    atm_plus_1_call = converter.get_desired_strike(expiry_as_dte, "CALL", 1, by="atm")
    atm_minus_1_put = converter.get_desired_strike(expiry_as_dte, "PUT", -1, by="atm")
    print(f"ATM+1 CALL strike: {atm_plus_1_call}")
    print(f"ATM-1 PUT strike: {atm_minus_1_put}")

    # Demonstrating percentage-based ATM relative strikes
    print("\nDemonstrating percentage-based ATM relative strikes:")
    atm_plus_1pct = converter.get_desired_strike(
        expiry_as_dte,
        "CALL",
        1,
        by="atm_percent",
    )
    atm_minus_1pct = converter.get_desired_strike(
        expiry_as_dte, "PUT", -1, by="atm_percent"
    )
    print(f"ATM+1% strike: {atm_plus_1pct}")
    print(f"ATM-1% strike: {atm_minus_1pct}")
