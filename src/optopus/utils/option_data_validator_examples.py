import pandas as pd
from option_data_validator import validate_option_data

if __name__ == "__main__":
    # Replace 'sample_option_data.parquet' with the actual path to your parquet file
    parquet_file_path = "/Users/traderHuy/Library/CloudStorage/OneDrive-Personal/Documents/optopus-dev/data/SPY_2024-09-06 15-30.parquet"
    try:
        # Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(parquet_file_path)

        # Validate the option data
        validated_df = validate_option_data(df)

        # Print the validated DataFrame
        print("Validated DataFrame:")
        print(validated_df)

    except FileNotFoundError:
        print(f"Error: File not found at {parquet_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
