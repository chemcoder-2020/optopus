import pandas as pd
import numpy as np


def compare_near_atm_prices(
    df: pd.DataFrame, target_dte: int, comparison_range: int = 3
):
    """
    Compares call and put prices for strikes near the money for a specific DTE.

    Args:
        df (pd.DataFrame): DataFrame containing the option chain data with columns like
                           'UNDERLYING_LAST', 'STRIKE', 'DTE', 'C_MARK', 'P_MARK'.
        target_dte (int): The specific Days Till Expiration (DTE) to analyze.
        comparison_range (int): How many strikes above/below ATM to compare (e.g., 3 means ATM+/-3).

    Returns:
        pd.DataFrame: A DataFrame showing the comparison results, or None if data is insufficient.
    """
    # Filter for the specific DTE
    df_filtered = df[df["intDTE"] == target_dte].copy()

    if df_filtered.empty:
        return None

    # Get the underlying price (use the first row's value for this expiration)
    underlying_price = df_filtered["UNDERLYING_LAST"].iloc[0]

    # Find the ATM strike (closest strike to the underlying price)
    df_filtered["strike_diff"] = abs(df_filtered["STRIKE"] - underlying_price)
    atm_strike = df_filtered.loc[df_filtered["strike_diff"].idxmin()]["STRIKE"]

    # Prepare data for easier lookup
    df_lookup = df_filtered.drop_duplicates(subset=["STRIKE"]).set_index("STRIKE")

    results = []

    # Iterate through the comparison range (0 is ATM, 1 is ATM+/-1, etc.)
    for n in range(comparison_range + 1):
        call_strike = atm_strike + n
        put_strike = atm_strike - n

        # Get ATM comparison (n=0)
        if n == 0:
            try:
                call_price = df_lookup.loc[atm_strike, "C_MARK"]
                put_price = df_lookup.loc[atm_strike, "P_MARK"]
                results.append(
                    {
                        "Comparison": f"ATM ({atm_strike})",
                        "Call Strike": atm_strike,
                        "Put Strike": atm_strike,
                        "Call Mark": call_price,
                        "Put Mark": put_price,
                        "Difference (C-P)": call_price - put_price,
                        "Pct Difference (C-P)": abs(call_price - put_price)
                        / ((put_price + call_price) / 2),
                    }
                )
            except KeyError:
                continue  # Skip if ATM strike itself is missing

        # Get OTM comparisons (n > 0)
        else:
            call_price = np.nan
            put_price = np.nan
            try:
                call_price = df_lookup.loc[call_strike, "C_MARK"]
            except KeyError:
                pass

            try:
                put_price = df_lookup.loc[put_strike, "P_MARK"]
            except KeyError:
                pass

            # Only add if we found both prices for the pair
            if not np.isnan(call_price) and not np.isnan(put_price):
                results.append(
                    {
                        "Comparison": f"ATM+{n} Call vs ATM-{n} Put",
                        "Call Strike": call_strike,
                        "Put Strike": put_strike,
                        "Call Mark": call_price,
                        "Put Mark": put_price,
                        "Difference (C-P)": call_price - put_price,
                        "Pct Difference (C-P)": abs(call_price - put_price)
                        / ((put_price + call_price) / 2),
                    }
                )

    if not results:
        return None

    return pd.DataFrame(results)
