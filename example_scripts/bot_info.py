import pickle
import os
from trade_manager import TradingManager
from typing import List
import pandas as pd
import pprint

pd.options.display.max_columns = 50


def load_trading_manager(pkl_path: str) -> TradingManager:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {pkl_path}")
    with open(pkl_path, "rb") as file:
        trading_manager = pickle.load(file)
    return trading_manager


def main():
    pkl_path = "trading_manager60dte.pkl"  # Update if your pickle file has a different name or path
    trading_manager = load_trading_manager(pkl_path)
    trading_manager.auth_refresh(
        client_id=os.getenv("SCHWAB_CLIENT_ID"),
        client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI"),
        token_file="token.json",
    )
    trading_manager.update_orders(pd.Timestamp.now(tz="America/New_York").floor("15min").tz_localize(None))
    # Get the orders DataFrame and print it
    orders_df = trading_manager.get_active_orders_dataframe()
    all_orders_df = trading_manager.get_orders_dataframe()
    total_pl = all_orders_df['Total P/L'].sum()
    closed_pl = all_orders_df[all_orders_df["Status"] == "Closed"]['Total P/L'].sum()
    trading_manager.freeze(pkl_path)

    print("-" * 50)
    print("Bot Name: ", pkl_path)
    print("Total P/L (commission included): " + str(total_pl))
    print("Closed P/L (commission included): " + str(closed_pl))
    print("=" * 50)
    print("\nActive Orders:\n")
    print(orders_df)
    print("\n")


if __name__ == "__main__":
    main()
