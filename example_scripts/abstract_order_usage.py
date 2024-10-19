import os
import dotenv
import pandas as pd
from schwab_order import SchwabOptionOrder
from abstract_order import AbstractOptionOrder
from option_spread import OptionStrategy

# Load environment variables
dotenv.load_dotenv()

# Sample option chain data
entry_df = pd.read_parquet(
    "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-10-04 15-15.parquet"
)

# Create a vertical spread strategy
vertical_spread = OptionStrategy.create_vertical_spread(
    symbol="SPY",
    option_type="PUT",
    long_strike="-2",
    short_strike="ATM",
    expiration="2024-11-29",
    contracts=1,
    entry_time="2024-10-04 15:15:00",
    option_chain_df=entry_df,
)

order = AbstractOptionOrder(
    option_strategy=vertical_spread,
    broker="Schwab",
)

# Create a SchwabOptionOrder instance
schwab_order = SchwabOptionOrder(
    client_id=os.getenv("SCHWAB_CLIENT_ID"),
    client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
    option_strategy=vertical_spread,
    token_filepath="token.json",
)

# Print order details
print("Schwab Option Order Details:")
print(f"Symbol: {schwab_order.symbol}")
print(f"Strategy Type: {schwab_order.strategy_type}")
print(f"Expiration: {schwab_order.legs[0].expiration}")
print(f"Long Strike: {schwab_order.legs[0].strike}")
print(f"Short Strike: {schwab_order.legs[1].strike}")
print(f"Contracts: {schwab_order.contracts}")
print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
print(f"Order Status: {schwab_order.order_status}")
print(f"Time: {schwab_order.current_time}")
print(f"DIT: {schwab_order.DIT}")
print(f"Net Premium: {schwab_order.net_premium}")

# Update the order with fresh quotes
schwab_order.update_order()

print("Schwab Option Order Details (After Update):")
print(f"Symbol: {schwab_order.symbol}")
print(f"Strategy Type: {schwab_order.strategy_type}")
print(f"Expiration: {schwab_order.legs[0].expiration}")
print(f"Long Strike: {schwab_order.legs[0].strike}")
print(f"Short Strike: {schwab_order.legs[1].strike}")
print(f"Contracts: {schwab_order.contracts}")
print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
print(f"Order Status: {schwab_order.order_status}")
print(f"Time: {schwab_order.current_time}")
print(f"DIT: {schwab_order.DIT}")
print(f"Net Premium: {schwab_order.net_premium}")

# Example of placing the entry order
# Note: This won't actually execute without valid credentials
# schwab_order.submit_entry()

# Example of placing the exit order
# Note: This won't actually execute without valid credentials
# schwab_order.submit_exit()

# Example of updating the order with new option chain data
# update_df = pd.read_parquet("path/to/your/updated_option_chain_data.parquet")
# schwab_order.update_order(update_df)
# print(f"Updated Order Status: {schwab_order.order_status}")
# print(f"Current Net Premium: {schwab_order.net_premium:.2f}")

# Example of canceling the order
# schwab_order.cancel()

# Example of modifying the order
# new_payload = {
#     "orderType": "LIMIT",
#     "price": 1.5,
#     "quantity": 1,
#     "duration": "DAY",
#     "orderStrategyType": "SINGLE",
#     "orderLegCollection": [
#         {
#             "instruction": "BUY_TO_CLOSE",
#             "quantity": 1,
#             "instrument": {
#                 "symbol": "SPY_111523P400",
#                 "assetType": "OPTION"
#             }
#         }
#     ]
# }
# schwab_order.modify(new_payload)
