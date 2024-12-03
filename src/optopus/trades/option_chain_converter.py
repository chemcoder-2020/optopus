import pandas as pd
from pandas import Timestamp, Timedelta
import datetime

class OptionChainConverter:
    """
    A class to handle conversions and selections from an option chain DataFrame.

    Methods:
        get_strike(symbol: str, option_chain_df: pd.DataFrame, strike_selector, option_type: str, reference_strike=None, expiration=None) -> float:
            Get the strike price based on the provided selector.

        get_expiration(option_chain_df: pd.DataFrame, expiration_input, entry_time: str) -> str:
            Get the expiration date based on the provided input.
    """

    @staticmethod
    def get_strike(
        symbol: str,
        option_chain_df: pd.DataFrame,
        strike_selector,
        option_type: str,
        reference_strike=None,
        expiration=None,  # Add expiration as a parameter
    ):
        """
        Get the strike price based on the provided selector.

        Args:
            symbol (str): The underlying asset symbol.
            option_chain_df (pd.DataFrame): The option chain data.
            strike_selector: The strike selection criteria.
            option_type (str): The option type ('CALL' or 'PUT').
            reference_strike (float, optional): A reference strike price for relative selection.
            expiration (str, optional): The expiration date for delta-based strike selection.

        Returns:
            float: The selected strike price.

        Raises:
            ValueError: If the strike selector is invalid.
        """
        if isinstance(strike_selector, (int, float)):
            return strike_selector
        elif strike_selector == "ATM":
            if option_type == "CALL":
                otm_calls = option_chain_df["STRIKE"].ge(
                    option_chain_df["UNDERLYING_LAST"]
                )
                option_data = option_chain_df[otm_calls].copy()
                option_data.sort_values(by="STRIKE", ascending=True, inplace=True)
            else:
                otm_puts = option_chain_df["STRIKE"].le(
                    option_chain_df["UNDERLYING_LAST"]
                )
                option_data = option_chain_df[otm_puts].copy()
                option_data.sort_values(by="STRIKE", ascending=False, inplace=True)

            strike = (
                option_data["STRIKE"].iloc[0]
                if int(option_data["STRIKE"].iloc[0])
                == float(option_data["STRIKE"].iloc[0])
                else option_data["STRIKE"].iloc[1]
            )

            return strike
        elif isinstance(strike_selector, str):
            if strike_selector.startswith(("+", "-")):
                strike_value = float(strike_selector)
                if abs(strike_value) < 1:
                    # Assume it's delta selection
                    if expiration is None:
                        raise ValueError(
                            "Expiration is required for delta-based strike selection"
                        )

                    target_dte = (
                        pd.to_datetime(expiration).tz_localize(None)
                        - pd.to_datetime(
                            option_chain_df["QUOTE_READTIME"].iloc[0]
                        ).tz_localize(None)
                    ).days

                    return OptionChainConverter._get_strike_from_delta(
                        symbol=symbol,
                        option_type=option_type,
                        target_delta=strike_value,
                        target_dte=target_dte,
                        option_chain_df=option_chain_df,
                    )
                else:
                    # Relative strike selection
                    if reference_strike is None:
                        raise ValueError(
                            "Reference strike is required for relative strike selection"
                        )
                    return reference_strike + strike_value
        else:
            raise ValueError(f"Invalid strike selector: {strike_selector}")

    @staticmethod
    def _get_strike_from_delta(
        symbol: str,
        option_type: str,
        target_delta: float,
        target_dte: int,
        option_chain_df: pd.DataFrame,
    ):
        """
        Get the strike price based on the target delta and DTE.

        Args:
            symbol (str): The underlying asset symbol.
            option_type (str): The option type ('CALL' or 'PUT').
            target_delta (float): The target delta.
            target_dte (int): The target DTE.
            option_chain_df (pd.DataFrame): The option chain data.

        Returns:
            float: The selected strike price.
        """
        option_chain_df["DTE"] = (
            pd.to_datetime(option_chain_df["EXPIRE_DATE"]).dt.tz_localize(None)
            - pd.to_datetime(
                option_chain_df["QUOTE_READTIME"].iloc[0]
            ).dt.tz_localize(None)
        ).dt.days

        valid_expirations = option_chain_df[option_chain_df["DTE"] == target_dte]

        if valid_expirations.empty:
            raise ValueError(f"No expiration found with DTE {target_dte}")

        valid_options = valid_expirations[valid_expirations["OPTION_TYPE"] == option_type]

        if valid_options.empty:
            raise ValueError(f"No {option_type} options found for DTE {target_dte}")

        valid_options["ABS_DELTA"] = abs(valid_options["DELTA"] - target_delta)
        closest_option = valid_options.loc[valid_options["ABS_DELTA"].idxmin()]

        return closest_option["STRIKE"]

    @staticmethod
    def get_expiration(
        option_chain_df: pd.DataFrame, expiration_input, entry_time: str
    ):
        """
        Get the expiration date based on the provided input.

        Args:
            option_chain_df (pd.DataFrame): The option chain data.
            expiration_input (str or int): The expiration date or target DTE.
            entry_time (str): The entry time for the strategy.

        Returns:
            str: The selected expiration date.

        Raises:
            ValueError: If no suitable expiration is found.
        """
        entry_date = pd.to_datetime(entry_time).tz_localize(None)

        if isinstance(expiration_input, str):
            target_date = pd.to_datetime(expiration_input).tz_localize(None)
            valid_expirations = option_chain_df[
                pd.to_datetime(option_chain_df["EXPIRE_DATE"]).dt.tz_localize(None) == target_date
            ]

            if valid_expirations.empty:
                raise ValueError(f"No expiration found for date {expiration_input}")

            return target_date.strftime("%Y-%m-%d")

        elif isinstance(expiration_input, (int, float)):
            target_dte = float(expiration_input)
            valid_expirations = option_chain_df[
                pd.to_datetime(option_chain_df["EXPIRE_DATE"]).dt.tz_localize(None) - entry_date >= Timedelta(days=target_dte)
            ]

            if valid_expirations.empty:
                raise ValueError(f"No expiration found with DTE >= {target_dte}")

            closest_expiration = valid_expirations.loc[
                valid_expirations["DTE"].idxmin(), "EXPIRE_DATE"
            ]
            return pd.to_datetime(closest_expiration).strftime("%Y-%m-%d")

        else:
            raise ValueError(
                "Invalid expiration input. Must be a date string or a number (DTE)."
            )

if __name__ == "__main__":
    # Load the test option chain DataFrame
    test_option_chain_df = pd.read_parquet("/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-06 15-30.parquet")

    # Define test parameters
    symbol = "SPY"
    strike_selector = "-0.4"  # Change this to test different selectors
    option_type = "CALL"  # Change this to test different option types
    expiration_input = 30  # Change this to test different expiration inputs
    entry_time = "2024-09-06 15:30:00"  # Change this to test different entry times

    # Get the strike price
    try:
        strike = OptionChainConverter.get_strike(
            symbol=symbol,
            option_chain_df=test_option_chain_df,
            strike_selector=strike_selector,
            option_type=option_type,
            expiration=expiration_input,
        )
        print(f"Selected Strike: {strike}")
    except ValueError as e:
        print(f"Error selecting strike: {e}")

    # Get the expiration date
    try:
        expiration = OptionChainConverter.get_expiration(
            option_chain_df=test_option_chain_df,
            expiration_input=expiration_input,
            entry_time=entry_time,
        )
        print(f"Selected Expiration: {expiration}")
    except ValueError as e:
        print(f"Error selecting expiration: {e}")
