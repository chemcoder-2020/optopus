import argparse
import os
import shutil
import configparser
from optopus.trades.option_manager import Config as OptopusConfig
from optopus.trades.exit_conditions import DefaultExitCondition
import pandas as pd

def create_directory(project_name):
    """Creates the project directory."""
    os.makedirs(project_name, exist_ok=True)

def generate_config_file(project_name, args):
    """Generates the config.py file."""
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "templates", "0dte-IB", "config.py"
    )
    config_path = os.path.join(project_name, "config.py")

    # Read the template file
    with open(template_path, "r") as f:
        template_content = f.read()

    # Parse the STRATEGY_PARAMS and BACKTESTER_CONFIG sections
    config = configparser.ConfigParser()
    config.read_string(template_content)

    # Update STRATEGY_PARAMS
    strategy_params = dict(config["STRATEGY_PARAMS"])
    if args.symbol:
        strategy_params["symbol"] = f'"{args.symbol}"'
    if args.dte:
        strategy_params["dte"] = args.dte
    if args.strike:
        strategy_params["strike"] = f'"{args.strike}"'
    if args.contracts:
        strategy_params["contracts"] = args.contracts
    if args.commission:
        strategy_params["commission"] = args.commission

    # Update BACKTESTER_CONFIG
    backtester_config = dict(config["BACKTESTER_CONFIG"])
    if args.initial_capital:
        backtester_config["initial_capital"] = args.initial_capital
    if args.max_positions:
        backtester_config["max_positions"] = args.max_positions
    if args.position_size:
        backtester_config["position_size"] = args.position_size

    # Format the updated sections back into the template
    updated_strategy_params = "\n".join(
        f"{key} = {value}" for key, value in strategy_params.items()
    )
    updated_backtester_config = "\n".join(
        f"{key} = {value}" for key, value in backtester_config.items()
    )

    # Replace the old sections with the updated ones
    config_content = template_content.replace(
        "\n".join(f"{key} = {value}" for key, value in dict(config["STRATEGY_PARAMS"]).items()),
        updated_strategy_params,
    )
    config_content = config_content.replace(
        "\n".join(f"{key} = {value}" for key, value in dict(config["BACKTESTER_CONFIG"]).items()),
        updated_backtester_config,
    )

    # Write the updated content to the new config.py
    with open(config_path, "w") as f:
        f.write(config_content)

def copy_template_files(project_name):
    """Copies template files to the project directory."""
    templates_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "templates", "0dte-IB"
    )
    files_to_copy = ["backtest.py", "entry_condition.py", "exit_condition.py", "backtest_cross_validate.py"]

    for file_name in files_to_copy:
        source_path = os.path.join(templates_dir, file_name)
        dest_path = os.path.join(project_name, file_name)
        shutil.copy2(source_path, dest_path)

def copy_optopus_files(project_name):
    """Copies necessary files from optopus to the project directory."""
    optopus_dir = os.path.join(os.path.dirname(__file__), "..")
    files_to_copy = [
        "backtest/base_backtest.py",
        "trades/exit_conditions.py",
        "trades/option_manager.py",
        "trades/option_spread.py",
        "trades/option_leg.py",
        "trades/entry_conditions.py",
        "trades/option_chain_converter.py",
        "utils/heapmedian.py",
    ]

    for file_path in files_to_copy:
        source_path = os.path.join(optopus_dir, file_path)
        dest_path = os.path.join(project_name, "optopus", file_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)

def main():
    parser = argparse.ArgumentParser(description="Set up a backtesting project.")
    parser.add_argument("project_name", help="Name of the backtesting project directory")
    parser.add_argument("--symbol", help="Underlying symbol for the strategy")
    parser.add_argument("--dte", type=int, help="Days to expiration")
    parser.add_argument("--strike", help="Strike selection method (e.g., 'ATM', 'ATM+1%')")
    parser.add_argument("--contracts", type=int, help="Number of contracts to trade")
    parser.add_argument("--commission", type=float, help="Commission per contract")
    parser.add_argument(
        "--initial_capital", type=float, help="Initial capital for the backtest"
    )
    parser.add_argument(
        "--max_positions", type=int, help="Maximum number of open positions"
    )
    parser.add_argument(
        "--position_size", type=float, help="Position size as a fraction of capital"
    )

    args = parser.parse_args()

    create_directory(args.project_name)
    generate_config_file(args.project_name, args)
    copy_template_files(args.project_name)
    copy_optopus_files(args.project_name)

    print(f"Backtesting project '{args.project_name}' created successfully!")

if __name__ == "__main__":
    main()
