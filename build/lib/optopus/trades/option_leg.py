import datetime
from loguru import logger
import pandas as pd
import numpy as np
from typing import Union


class OptionLeg:
    """
    Represents a single option leg in an options trading strategy.

    Attributes:
        symbol (str): The underlying asset symbol.
        option_type (str): The type of option ('CALL' or 'PUT').
        strike (float): The strike price of the option.
        expiration (datetime): The expiration date of the option.
        contracts (int): The number of contracts.
        entry_time (datetime): The entry time for the option position.
        position_side (str): The side of the position ('BUY' or 'SELL').
        current_time (datetime): The current time of the option.
        entry_price (float): The entry price of the option.
        entry_underlying_last (float): The underlying last price at entry.
        current_price (float): The current price of the option.
        current_mark (float): The current mark price of the option.
        current_last (float): The current last price of the option.
        current_delta (float): The current delta of the option.
        underlying_last (float): The current underlying last price.
        underlying_diff (float): The difference between current and entry underlying last price.
        is_itm (bool): Whether the option is in-the-money.
        price_diff (float): The difference between current and entry price.
        pl (float): The profit/loss of the option.
        dte (float): The days to expiration of the option.
        commission (float): The commission per contract.
    """

    def __init__(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        position_side: str,
        commission: float = 0.5,  # Default commission per contract
    ):
        """
        Initialize an OptionLeg instance.

        Args:
            symbol (str): The underlying asset symbol.
            option_type (str): The type of option ('CALL' or 'PUT').
            strike (float): The strike price of the option.
            expiration: The expiration date of the option.
            contracts (int): The number of contracts.
            entry_time (str): The entry time for the option position.
            option_chain_df (pd.DataFrame): The option chain data.
            position_side (str): The side of the position ('BUY' or 'SELL').
            commission (float): The commission per contract.
        """

        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("Option type must be CALL or PUT")
        if not isinstance(strike, (int, float)):
            raise ValueError("Strike must be a number")
        if not isinstance(contracts, int):
            raise ValueError("Contracts must be an integer")
        if not isinstance(option_chain_df, pd.DataFrame):
            raise ValueError("Option chain df must be a pandas DataFrame")
        if position_side not in ["BUY", "SELL"]:
            raise ValueError("Position side must be BUY or SELL")
        if not isinstance(commission, (int, float)):
            raise ValueError("Commission must be a number")

        self.symbol = symbol
        self.option_type = option_type
        self.strike = strike
        self.expiration = pd.to_datetime(expiration).tz_localize(None)
        self.contracts = contracts
        self.entry_time = pd.to_datetime(entry_time).tz_localize(None)
        self.position_side = position_side.upper()
        self.current_time = None
        self.entry_price = None
        self.entry_underlying_last = None
        self.current_price = None
        self.current_mark = None
        self.current_last = None
        self.current_bid = None
        self.current_ask = None
        self.current_delta = None
        self.underlying_last = None
        self.underlying_diff = None
        self.is_itm = None
        self.price_diff = 0
        self.pl = 0
        self.dte = None
        self.commission = commission

        self.update(entry_time, option_chain_df, is_entry=True)

    def update(self, current_time, option_chain_df, is_entry=False):
        """
        Update the option leg with new market data.

        Args:
            current_time: The current time to update the option data.
            option_chain_df (pd.DataFrame): The updated option chain data.
            is_entry (bool): Whether this update is the entry point.

        Raises:
            ValueError: If the current_time format is invalid or if QUOTE_READTIME is inconsistent.
        """
        # Convert current_time to pd.Timestamp once
        if isinstance(current_time, str):
            current_datetime = pd.to_datetime(current_time).tz_localize(None)
        elif isinstance(current_time, pd.Timestamp):
            current_datetime = current_time.tz_localize(None)
        elif isinstance(current_time, datetime.datetime):
            current_datetime = pd.to_datetime(current_time).tz_localize(None)
        else:
            raise ValueError(
                "Invalid current_time format. Must be str, pd.Timestamp, or datetime."
            )

        self.current_time = current_datetime

        # Ensure QUOTE_READTIME is timezone-naive and rounded to 15 minutes
        quote_readtime = (
            option_chain_df["QUOTE_READTIME"].iloc[0].round("15min").tz_localize(None)
        )
        rounded_current_time = current_datetime.round("15min")

        if quote_readtime != rounded_current_time:
            raise ValueError(
                f"QUOTE_READTIME ({quote_readtime}) does not match rounded current_time ({rounded_current_time})"
            )

        # Filter the option chain for the specific option
        option_data = option_chain_df[
            (option_chain_df["STRIKE"] == self.strike)
            & (option_chain_df["EXPIRE_DATE"] == self.expiration)
        ]

        if not option_data.empty:
            prefix = "C_" if self.option_type.upper() == "CALL" else "P_"

            # Optimize MARK calculation
            mark_key = f"{prefix}MARK"
            bid_key = f"{prefix}BID"
            ask_key = f"{prefix}ASK"
            last_key = f"{prefix}LAST"
            delta_key = f"{prefix}DELTA"
            itm_key = f"{prefix}ITM"

            if bid_key in option_data:
                self.current_bid = option_data[bid_key].iloc[0]

            if ask_key in option_data:
                self.current_ask = option_data[ask_key].iloc[0]

            if mark_key in option_data:
                self.current_mark = option_data[mark_key].iloc[0]
            else:
                bid = (
                    option_data[bid_key].iloc[0]
                    if bid_key in option_data.columns
                    else None
                )
                ask = (
                    option_data[ask_key].iloc[0]
                    if ask_key in option_data.columns
                    else None
                )

                if pd.notna(bid) and pd.notna(ask):
                    self.current_mark = (bid + ask) / 2
                else:
                    logger.warning(f"Unable to calculate mark price for {self}")
                    self.current_mark = np.nan

            self.current_last = (
                option_data[last_key].iloc[0]
                if last_key in option_data.columns
                else None
            )
            self.current_delta = (
                option_data[delta_key].iloc[0]
                if delta_key in option_data.columns
                else None
            )
            self.underlying_last = (
                option_data["UNDERLYING_LAST"].iloc[0]
                if "UNDERLYING_LAST" in option_data.columns
                else None
            )
            self.is_itm = (
                option_data[itm_key].iloc[0] if itm_key in option_data.columns else None
            )

            self.current_price = (
                self.current_mark if pd.notna(self.current_mark) else np.nan
            )

            if is_entry:
                self.entry_price = self.current_mark
                self.entry_underlying_last = self.underlying_last
            else:
                self.underlying_diff = self.underlying_last - self.entry_underlying_last
                self.price_diff = self.current_price - self.entry_price
                self.pl = self.calculate_pl()

            # Calculate DTE
            self.dte = calculate_dte(self.expiration, current_datetime)
        else:
            logger.warning(f"No matching option found in the chain for {self}")
            if is_entry:
                self.entry_price = np.nan
                self.entry_underlying_last = np.nan
            self.dte = calculate_dte(self.expiration, current_datetime)

    def calculate_pl(self):
        """
        Calculate the profit/loss for the option leg.

        Returns:
            float: The calculated profit/loss.
        """
        multiplier = 1 if self.position_side == "BUY" else -1
        total_commission = self.calculate_total_commission()
        return (
            multiplier * self.price_diff * self.contracts * 100 - total_commission
        )  # Assuming each contract is for 100 shares

    def calculate_total_commission(self):
        """
        Calculate the total commission for the option leg.

        Returns:
            float: The total commission.
        """
        return self.commission * self.contracts * 2  # Opening and closing

    def update_entry_price(self, new_price: float):
        """Modify the entry price of the option leg. Helpful for updating entry prices after actual trading order is filled."""
        self.entry_price = new_price
        self.price_diff = self.current_price - self.entry_price
        self.pl = self.calculate_pl()

    def __repr__(self):
        return f"OptionLeg(symbol={self.symbol}, option_type={self.option_type}, strike={self.strike}, expiration={self.expiration}, contracts={self.contracts}, entry_price={self.entry_price}, current_price={self.current_price}, current_mark={self.current_mark}, current_last={self.current_last}, current_delta={self.current_delta}, entry_underlying_last={self.entry_underlying_last}, underlying_last={self.underlying_last}, underlying_diff={self.underlying_diff}, is_itm={self.is_itm}, price_diff={self.price_diff}, pl={self.pl}, position_side={self.position_side}, dte={self.dte}, commission={self.commission})"

    @classmethod
    def from_delta_and_dte(
        cls,
        symbol: str,
        option_type: str,
        target_delta: Union[str, float],  # Target delta,
        target_dte: float,
        contracts: int,
        entry_time: str,
        option_chain_df: pd.DataFrame,
        position_side: str,
        max_dte: float = None,
        reference_strike: float = None,
        commission: float = 0.5,  # Default commission per contract
    ):
        """
        Create an OptionLeg instance based on target delta and days to expiration (DTE).

        Args:
            symbol (str): The underlying asset symbol.
            option_type (str): The type of option ('CALL' or 'PUT').
            target_delta (float or str): The target delta for the option or 'ATM' for at-the-money.
            target_dte (float): The target days to expiration.
            entry_time (str): The entry time for the option position.
            contracts (int): The number of contracts.
            option_chain_df (pd.DataFrame): The option chain data.
            position_side (str): The side of the position ('BUY' or 'SELL').
            max_dte (float, optional): The maximum allowed days to expiration.
            reference_strike (float, optional): A reference strike price for relative selection.
            commission (float): The commission per contract.

        Returns:
            OptionLeg: An instance of OptionLeg matching the specified criteria.
        """
        # Ensure option_type is uppercase
        option_type = option_type.upper()

        # Filter the option chain for the correct option type
        prefix = "C_" if option_type == "CALL" else "P_"
        filtered_df = option_chain_df[option_chain_df[f"{prefix}DELTA"].notna()]

        # Calculate DTE for each expiration
        current_date = pd.to_datetime(entry_time).tz_localize(None)

        # Filter by DTE
        if max_dte:
            filtered_df = filtered_df[filtered_df["DTE"].between(target_dte, max_dte)]
        else:
            filtered_df = filtered_df.loc[filtered_df["DTE"].ge(target_dte)]

        # Sort by DTE to get the closest expiration
        expiration_df = filtered_df[filtered_df["DTE"].eq(filtered_df["DTE"].min())]

        # Determine the comparison method based on option type and position side
        if option_type == "CALL":
            comparison = "higher" if position_side == "BUY" else "lower"
        else:  # PUT
            comparison = "lower" if position_side == "BUY" else "higher"

        # Find the strike with the closest delta using calculate_closest_match
        closest_match = cls.calculate_closest_match(
            subset=expiration_df,
            delta_col=f"{prefix}DELTA",
            target_delta=target_delta,
            comparison=comparison,
            option_type=option_type.lower(),
            reference_strike=reference_strike,
        )

        if closest_match is None:
            raise ValueError("No matching option found for the given delta and DTE.")

        closest_strike = closest_match["STRIKE"]
        closest_expiration = closest_match["EXPIRE_DATE"]

        # Create and return the OptionLeg instance
        return cls(
            symbol=symbol,
            option_type=option_type,
            strike=closest_strike,
            expiration=closest_expiration,
            contracts=contracts,
            entry_time=entry_time,
            option_chain_df=option_chain_df,
            position_side=position_side,
            commission=commission,
        )

    @staticmethod
    def calculate_closest_match(
        subset,
        delta_col,
        target_delta,
        comparison="closest",
        option_type=None,
        reference_strike=None,
    ):
        """
        Calculate the closest match to the target delta in the given subset of options.

        Args:
            subset (pd.DataFrame): A subset of the option chain.
            delta_col (str): The name of the delta column.
            target_delta (float or str): The target delta or 'ATM' for at-the-money.
            comparison (str): The comparison method ('closest', 'lower', or 'higher').
            option_type (str, optional): The option type ('call' or 'put').
            reference_strike (float, optional): A reference strike price for relative selection.

        Returns:
            pd.Series: The row from the subset that best matches the criteria.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        subset = subset.copy()

        if isinstance(target_delta, (float, int)):
            if abs(target_delta) <= 1:
                subset["abs_delta_diff"] = (subset[delta_col] - target_delta).abs()

                # Apply comparison filter before sorting
                if comparison == "lower":
                    subset = subset[subset[delta_col] <= target_delta]
                elif comparison == "higher":
                    subset = subset[subset[delta_col] >= target_delta]

                if not subset.empty:
                    return subset.nsmallest(1, "abs_delta_diff").iloc[0]
                else:
                    return None
            else:
                if reference_strike:
                    target_strike = reference_strike + target_delta
                    subset["abs_strike_diff"] = (subset["STRIKE"] - target_strike).abs()
                    subset = subset.nsmallest(1, "abs_strike_diff")
                    return subset.iloc[0] if not subset.empty else None
                else:
                    raise ValueError(
                        "reference_strike must be provided for relative strike selection"
                    )
        elif (
            isinstance(target_delta, str)
            and target_delta.upper() == "ATM"
            and option_type
        ):
            underlying_price = subset["UNDERLYING_LAST"].iloc[0]
            if option_type.lower() == "call":
                subset = subset[subset["STRIKE"] >= underlying_price]
            else:  # put
                subset = subset[subset["STRIKE"] <= underlying_price]
            subset["abs_strike_diff"] = (subset["STRIKE"] - underlying_price).abs()
            return (
                subset.nsmallest(1, "abs_strike_diff").iloc[0]
                if not subset.empty
                else None
            )
        else:
            raise ValueError("Invalid target_delta or option_type")

    def conflicts_with(self, other_leg):
        """
        Check if this option leg conflicts with another option leg.

        Args:
            other_leg (OptionLeg): The other option leg to compare with.

        Returns:
            bool: True if there's a conflict, False otherwise.
        """
        return (
            self.symbol == other_leg.symbol
            and self.option_type == other_leg.option_type
            and self.strike == other_leg.strike
            and self.expiration == other_leg.expiration
        )

    @property
    def schwab_symbol(self):
        if isinstance(self.expiration, pd.Timestamp):
            expiration = self.expiration.strftime("%y%m%d")
        elif isinstance(self.expiration, str):
            if len(self.expiration) != 6:
                expiration = pd.Timestamp(self.expiration).strftime("%y%m%d")
        option_symbol = f"{self.symbol.ljust(6)}{expiration}{self.option_type[0]}{str(int(self.strike * 1000)).zfill(8)}"
        return option_symbol


def calculate_dte(expiration_date, current_date) -> float:
    """
    Calculate the days to expiration (DTE) for an option.

    Args:
        expiration_date: The expiration date of the option.
        current_date: The current date.

    Returns:
        float: The calculated days to expiration.

    Raises:
        ValueError: If the date formats are invalid.
    """
    if isinstance(expiration_date, str):
        expiration_datetime = pd.to_datetime(expiration_date).tz_localize(None)
    elif isinstance(expiration_date, pd.Timestamp):
        expiration_datetime = expiration_date.tz_localize(None)
    elif isinstance(expiration_date, datetime.datetime):
        expiration_datetime = pd.to_datetime(expiration_date).tz_localize(None)
    else:
        raise ValueError(
            "Invalid expiration_date format. Must be str, pd.Timestamp, or datetime."
        )

    if isinstance(current_date, str):
        current_datetime = pd.to_datetime(current_date).tz_localize(None)
    elif isinstance(current_date, pd.Timestamp):
        current_datetime = current_date.tz_localize(None)
    elif isinstance(current_date, datetime.datetime):
        current_datetime = pd.to_datetime(current_date).tz_localize(None)
    else:
        raise ValueError(
            "Invalid current_date format. Must be str, pd.Timestamp, or datetime."
        )

    expiration_datetime = expiration_datetime.replace(
        hour=16, minute=0, second=0, microsecond=0
    )
    return (expiration_datetime - current_datetime).total_seconds() / (60 * 60 * 24)
