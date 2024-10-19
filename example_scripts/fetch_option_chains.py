import os
from schwab_data import SchwabData

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()

def fetch_option_chains(ticker, output_folders):
    """
    Fetch option chains for a specified ticker and save them to output folders.

    Parameters:
        ticker (str): The ticker symbol for which to fetch the option chains.
        output_folders (list): The folders where the processed option chains will be saved.
    """
    # Initialize SchwabData with environment variables
    client_id = os.getenv("SCHWAB_CLIENT_ID")
    client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    schwab_data = SchwabData(client_id, client_secret)

    # Fetch the option chain and save it to the output folder
    data = schwab_data.get_option_chain(symbol=ticker, output_folders=output_folders)
    return data


if __name__ == "__main__":
    ticker = "SPY"
    output_folders = ["/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024", "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/ORATS/SPY/by_bar", "/Users/traderHuy/Downloads/SPY option backtest analysis/OptionDX/SPY/by_day/by_bar"]
    data = fetch_option_chains(ticker, output_folders)
