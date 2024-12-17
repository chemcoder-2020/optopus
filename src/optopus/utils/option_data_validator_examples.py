import pandas as pd
from option_data_validator import (
    validate_option_data,
    EXPECTED_OPTION_DATA_SCHEMA,
    OPTIONAL_OPTION_DATA_SCHEMA,
)

def create_modified_dataframe(original_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Creates a modified DataFrame based on the original DataFrame for testing different scenarios.

    Args:
        original_df: The original DataFrame.
        scenario: The scenario to test ('missing_required', 'incorrect_required_type', 'missing_optional', 'incorrect_optional_type').

    Returns:
        A modified DataFrame for the specified scenario.
    """
    df = original_df.copy()
    if scenario == "missing_required":
        # Remove a required column
        if "STRIKE" in df.columns:
            df = df.drop(columns=["STRIKE"])
    elif scenario == "incorrect_required_type":
        # Change the data type of a required column
        if "C_BID" in df.columns:
            df["C_BID"] = df["C_BID"].astype(str)
    elif scenario == "missing_optional":
        # Remove an optional column
        if "C_LAST" in df.columns:
            df = df.drop(columns=["C_LAST"])
    elif scenario == "incorrect_optional_type":
        # Change the data type of an optional column
        if "P_OI" in df.columns:
            df["P_OI"] = df["P_OI"].astype(float)
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
    return df

if __name__ == "__main__":
    # Replace with the actual path to your parquet file
    parquet_file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-06 15-30.parquet"

    try:
        # Read the parquet file into a pandas DataFrame
        original_df = pd.read_parquet(parquet_file_path)

        # Scenario 1: Original DataFrame (no modifications)
        print("Scenario 1: Original DataFrame")
        validated_df = validate_option_data(original_df)
        print("Validated DataFrame:\n", validated_df.head())
        # Assertions for original DataFrame
        for col, dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
            assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
            assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"

        # Scenario 2: Missing required column
        print("\nScenario 2: Missing required column")
        modified_df = create_modified_dataframe(original_df, "missing_required")
        validated_df = validate_option_data(modified_df)
        print("Validated DataFrame:\n", validated_df.head())
        # Assertions for missing required column
        for col, dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
            assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
            assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"

        # Scenario 3: Incorrect data type for a required column
        print("\nScenario 3: Incorrect data type for a required column")
        modified_df = create_modified_dataframe(original_df, "incorrect_required_type")
        validated_df = validate_option_data(modified_df)
        print("Validated DataFrame:\n", validated_df.head())
        # Assertions for incorrect data type of a required column
        for col, dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
            assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
            assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"

        # Scenario 4: Missing optional column
        print("\nScenario 4: Missing optional column")
        modified_df = create_modified_dataframe(original_df, "missing_optional")
        validated_df = validate_option_data(modified_df)
        print("Validated DataFrame:\n", validated_df.head())
        # Assertions for missing optional column (should still pass if missing)
        for col, dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
            assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
            assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"

        # Scenario 5: Incorrect data type for an optional column
        print("\nScenario 5: Incorrect data type for an optional column")
        modified_df = create_modified_dataframe(original_df, "incorrect_optional_type")
        validated_df = validate_option_data(modified_df)
        print("Validated DataFrame:\n", validated_df.head())
        # Assertions for incorrect data type of an optional column
        for col, dtype in EXPECTED_OPTION_DATA_SCHEMA.items():
            assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
            assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"
        for col, dtype in OPTIONAL_OPTION_DATA_SCHEMA.items():
            if col in modified_df.columns:
                assert col in validated_df.columns, f"Column {col} is missing in the validated DataFrame"
                assert validated_df[col].dtype == dtype, f"Column {col} has incorrect data type in the validated DataFrame"

    except FileNotFoundError:
        print(f"Error: File not found at {parquet_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
