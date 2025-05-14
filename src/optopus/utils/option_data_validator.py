import pandas as pd
import warnings

# Required columns
EXPECTED_OPTION_DATA_SCHEMA = {
    "STRIKE": "float64",
    "C_BID": "float64",
    "C_ASK": "float64",
    "C_MARK": "float64",
    "C_DELTA": "float64",
    "P_BID": "float64",
    "P_ASK": "float64",
    "P_MARK": "float64",
    "P_DELTA": "float64",
    "EXPIRE_DATE": "datetime64[ns]",
    "DTE": "float64",
    "intDTE": "int64",
    "UNDERLYING_LAST": "float64",
    "QUOTE_READTIME": "datetime64[ns, America/New_York]",
    "QUOTE_TIME_HOURS": "float64",
}

# Optional columns
OPTIONAL_OPTION_DATA_SCHEMA = {
    "C_LAST": "float64",
    "C_IV": "float64",
    "C_GAMMA": "float64",
    "C_THETA": "float64",
    "C_VEGA": "float64",
    "C_RHO": "float64",
    "C_OI": "int64",
    "C_ITM": "bool",
    "P_LAST": "float64",
    "P_IV": "float64",
    "P_GAMMA": "float64",
    "P_THETA": "float64",
    "P_VEGA": "float64",
    "P_RHO": "float64",
    "P_OI": "int64",
    "P_ITM": "bool",
    "INTEREST_RATE": "float64",
}


def _add_missing_columns(df: pd.DataFrame, expected_schema: dict) -> pd.DataFrame:
    """Adds missing columns to the DataFrame with default values."""
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            if dtype == "float64":
                df[col] = 0.0
            elif dtype == "int64":
                df[col] = 0
            elif dtype == "bool":
                df[col] = False
            elif dtype == "datetime64[ns]":
                df[col] = pd.NaT
            elif dtype == "datetime64[ns, America/New_York]":
                df[col] = pd.NaT
            else:
                df[col] = ""
    return df


def _convert_column_type(df: pd.DataFrame, col: str, dtype: str) -> pd.DataFrame:
    """Attempts to convert a column to the specified data type."""
    try:
        if dtype.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert column '{col}' to type '{dtype}': {e}")
    return df


def validate_option_data(df: pd.DataFrame, column_aliases: dict = None) -> pd.DataFrame:
    """
    Validates the option data DataFrame against the expected schema.

    Args:
        df: The option data DataFrame.
        column_aliases: A dictionary mapping expected column names to their aliases.

    Returns:
        The validated DataFrame.

    Raises:
        ValueError: If the DataFrame cannot be validated.
    """
    # Rename columns using aliases if provided
    if column_aliases:
        # Reverse the alias mapping to rename columns from aliases to expected names
        reverse_aliases = {v: k for k, v in column_aliases.items()}
        df = df.rename(columns=reverse_aliases)

    # Check for missing required columns
    missing_cols = set(EXPECTED_OPTION_DATA_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        warnings.warn(
            f"Missing required columns: {missing_cols}. Attempting to add them with default values."
        )
        df = _add_missing_columns(df, EXPECTED_OPTION_DATA_SCHEMA)

    # Check data types of required columns
    for col, expected_dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                warnings.warn(
                    f"Required column '{col}' has incorrect data type: expected '{expected_dtype}', found '{actual_dtype}'. Attempting to convert."
                )
                df = _convert_column_type(df, col, expected_dtype)

    # Check data types of optional columns if they exist
    for col, expected_dtype in OPTIONAL_OPTION_DATA_SCHEMA.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                warnings.warn(
                    f"Optional column '{col}' has incorrect data type: expected '{expected_dtype}', found '{actual_dtype}'. Attempting to convert."
                )
                df = _convert_column_type(df, col, expected_dtype)

    return df
