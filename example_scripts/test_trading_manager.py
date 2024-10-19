import os
import pandas as pd
from trade_manager import TradingManager
from schwab_order import SchwabOptionOrder
from option_manager import BacktesterConfig
from option_spread import OptionStrategy

# Load test data
entry_df = pd.read_parquet(
    "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-10-07 09-45.parquet"
)
update_df = pd.read_parquet(
    "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-10-07 15-45.parquet"
)

print("\nRunning TradingManager tests:")

# Test 1: Initialization
print("\nTest 1: Initialization")
config = BacktesterConfig(
    initial_capital=10000,
    max_positions=5,
    max_positions_per_day=1,
    max_positions_per_week=None,
    position_size=0.05,
    ror_threshold=0,
    gain_reinvesting=False,
)
manager = TradingManager(config)
assert manager.capital == 10000, "Initial capital not set correctly"
assert manager.config.max_positions == 5, "Max positions not set correctly"
print("Test 1 passed: TradingManager initialized correctly")

# Test 2: Adding an order
print("\nTest 2: Adding an order")
vertical_spread = OptionStrategy.create_vertical_spread(
    symbol="SPY",
    option_type="PUT",
    long_strike="-1",
    short_strike="ATM",
    expiration="2024-11-22",
    contracts=1,
    entry_time="2024-10-07 09:45:00",
    option_chain_df=entry_df,
)
order = SchwabOptionOrder(
    client_id=os.getenv("SCHWAB_CLIENT_ID"),
    client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
    option_strategy=vertical_spread,
    token_file="token.json",
)
assert manager.add_order(order), "Failed to add valid order"
assert len(manager.active_trades) == 1, "Active trades not updated correctly"
print("Test 2 passed: Order added successfully")

# Test 3: Updating orders
print("\nTest 3: Updating orders")
manager.update_orders("2024-10-07 15:45:00", update_df)
assert len(manager.active_trades) == 1, "Active trades not updated correctly"
print("Test 3 passed: Orders updated successfully")


print("\nAll TradingManager tests completed!")
