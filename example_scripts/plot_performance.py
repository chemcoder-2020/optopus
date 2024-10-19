import pickle
import matplotlib.pyplot as plt
import os
from trade_manager import TradingManager
from typing import List
import pandas as pd

def load_trading_manager(pkl_path: str) -> TradingManager:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {pkl_path}")
    with open(pkl_path, 'rb') as file:
        trading_manager = pickle.load(file)
    return trading_manager

def extract_trade_performance(trading_manager: TradingManager):
    performance_data = pd.DataFrame(trading_manager.performance_data)
    return performance_data

def plot_cumulative_pnl(performance_data: List[dict]):
    sorted_data = sorted(performance_data, key=lambda x: x['Exit Time'])
    cumulative_pnl = []
    times = []
    total = 0
    for data in sorted_data:
        total += data['P/L']
        cumulative_pnl.append(total)
        times.append(data['Exit Time'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, cumulative_pnl, marker='o')
    plt.xlabel('Exit Time')
    plt.ylabel('Cumulative P/L')
    plt.title('Cumulative Profit and Loss Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_pnl.png')
    plt.show()

def plot_trade_distribution(performance_data: List[dict]):
    pnl_values = [data['P/L'] for data in performance_data]
    plt.figure(figsize=(10, 5))
    plt.hist(pnl_values, bins=20, edgecolor='black')
    plt.xlabel('Profit and Loss')
    plt.ylabel('Number of Trades')
    plt.title('Distribution of Trade P/L')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pnl_distribution.png')
    plt.show()

def main():
    client_id = os.getenv("SCHWAB_CLIENT_ID")
    client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    redirect_uri = os.getenv("SCHWAB_REDIRECT_URI")
    token_file = os.getenv("SCHWAB_TOKEN_FILE", "token.json")
    pkl_path = 'trading_manager60dte.pkl'  # Update if your pickle file has a different name or path
    trading_manager = load_trading_manager(pkl_path)
    trading_manager.auth_refresh(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, token_file=token_file)
    performance_data = extract_trade_performance(trading_manager)
    
    if not performance_data:
        print("No closed trades to display.")
        return
    
    plot_cumulative_pnl(performance_data)
    plot_trade_distribution(performance_data)

if __name__ == "__main__":
    main()
