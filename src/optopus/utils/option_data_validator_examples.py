import pandas as pd
from optopus.utils.option_data_validator import validate_option_data

if __name__ == "__main__":
    # Replace 'sample_option_data.parquet' with the actual path to your parquet file
    parquet_file_path = "sample_option_data.parquet"
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
