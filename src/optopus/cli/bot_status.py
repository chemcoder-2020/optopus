import os
import sys
import dill
import argparse
import pandas as pd
from loguru import logger
import glob

# from pathlib import Path # Unused import

# Set up logging
# Ensure log file path is appropriate, maybe relative to script or user home
log_file = os.path.join(os.path.dirname(__file__), "bot_status.log")
logger.add(log_file, rotation="10 MB", retention="60 days", compression="zip")

pd.options.display.max_columns = 50


def check_available_bots():
    """
    Recursively check for any pkl files and report the name of their
    immediate parent folder.
    """
    # Consider making the search path configurable or more specific
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    # Filter out empty strings that might result from pkl files in root
    available_bots = {
        os.path.basename(os.path.dirname(pkl)) for pkl in pkl_files if os.path.basename(os.path.dirname(pkl))
    }
    logger.info(f"Found available bots: {available_bots}")
    return available_bots


def load_bot(pkl_path: str):
    """
    Load a bot from a pickle file.

    Args:
        pkl_path (str): Path to the pickle file (e.g., bot_name/trading_manager_bot_name.pkl).

    Returns:
        The loaded bot instance.
    """
    abs_pkl_path = os.path.abspath(pkl_path)
    if not os.path.exists(abs_pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {abs_pkl_path}")

    # Get the directory containing the pickle file
    pkl_dir = os.path.dirname(abs_pkl_path)

    original_sys_path = sys.path.copy()
    path_added = False

    try:
        # Add the directory to Python path if needed for unpickling
        if pkl_dir not in sys.path:
            logger.debug(f"Temporarily adding {pkl_dir} to sys.path for unpickling.")
            sys.path.insert(0, pkl_dir)
            path_added = True

        # Load the bot directly using the absolute path
        with open(abs_pkl_path, "rb") as file:
            bot = dill.load(file)
            logger.info(f"Successfully loaded bot from {abs_pkl_path}")

        return bot
    except Exception as e:
        logger.exception(f"Failed to load bot from {abs_pkl_path}: {e}")
        raise  # Re-raise the exception after logging
    finally:
        # Restore sys.path if it was modified
        if path_added and pkl_dir in sys.path:
            logger.debug(f"Removing {pkl_dir} from sys.path.")
            # Check if it's still the first element before removing
            if sys.path and sys.path[0] == pkl_dir:
                sys.path.pop(0)
            else:
                # Fallback in case something else modified sys.path
                try:
                    sys.path.remove(pkl_dir)
                except ValueError:
                    logger.warning(
                        f"Could not remove {pkl_dir} from sys.path "
                        "as it was not found."
                    )

        # Ensure sys.path is fully restored if other changes occurred
        # This is a safety net.
        if sys.path != original_sys_path and not path_added:
            logger.warning(
                "sys.path was modified unexpectedly during bot load. "
                "Attempting restore."
            )
            sys.path = original_sys_path


def update_exit(bot, **kwargs):
    """
    Update the exit scheme for each active order.

    Args:
        bot: The bot instance to update.
        **kwargs: Keyword arguments to pass to the exit scheme update method.
    """
    if not hasattr(bot, "active_orders"):
        logger.warning("Bot object does not have 'active_orders' attribute.")
        return

    for order in bot.active_orders:
        if hasattr(order, "exit_scheme") and callable(
            getattr(order.exit_scheme, "update", None)
        ):
            try:
                order.exit_scheme.update(**kwargs)
                logger.info(f"Updated exit scheme for order {order.id}")
            except Exception as e_update:
                logger.error(
                    f"Failed to update exit scheme for order {order.id}: {e_update}"
                )


def check_bot_status(bot):
    """
    Check the status of a bot.

    Args:
        bot: The bot instance to check.

    Returns:
        A dictionary containing the bot's status. Returns None if status check fails.
    """
    try:
        bot.update_orders()
    except Exception as e_update:
        logger.warning(
            f"Initial order update failed: {e_update}. Attempting auth refresh."
        )
        try:
            if hasattr(bot, "auth_refresh") and callable(bot.auth_refresh):
                bot.auth_refresh()
                bot.update_orders()
                logger.info("Successfully updated orders after auth refresh.")
            else:
                logger.error("Bot has no callable 'auth_refresh' method.")
                raise e_update  # Re-raise original error if no refresh possible
        except Exception as e_refresh:
            logger.exception(
                f"Order update failed even after auth refresh attempt: {e_refresh}"
            )
            return None  # Indicate failure

    try:
        orders_df = bot.get_active_orders_dataframe()
        if orders_df is not None:
            orders_df.sort_values(by="Entry Time", inplace=True)
        else:
            orders_df = pd.DataFrame()  # Ensure it's a DataFrame

        all_orders_df = bot.get_orders_dataframe()
        if all_orders_df is not None:
            all_orders_df.sort_values(by="Exit Time", inplace=True)
        else:
            all_orders_df = pd.DataFrame()  # Ensure it's a DataFrame

        allocation = getattr(bot, "allocation", 0)
        risk = 0
        if hasattr(bot, "active_orders"):
            risk = sum(
                trade.get_required_capital()
                for trade in bot.active_orders
                if hasattr(trade, "get_required_capital")
            )

        available_to_trade = getattr(bot, "available_to_trade", 0)

        total_pl = (
            all_orders_df["Total P/L"].sum() if "Total P/L" in all_orders_df else 0
        )
        closed_pl = 0
        if "Status" in all_orders_df and "Total P/L" in all_orders_df:
            closed_orders = all_orders_df[all_orders_df["Status"] == "CLOSED"]
            closed_pl = closed_orders["Total P/L"].sum()

        # Performance calculations
        total_pl_change_today = 0
        closed_pl_change_today = 0
        total_pl_change_MTD = 0
        closed_pl_change_MTD = 0

        if hasattr(bot, "performance_data") and bot.performance_data:
            try:
                perf_df = pd.DataFrame(bot.performance_data)
                if "time" in perf_df.columns:
                    perf_df["time"] = pd.to_datetime(perf_df["time"])
                    perf_df = perf_df.set_index("time")

                    # Daily performance
                    perf_data_daily = perf_df.resample("D").last()
                    if "total_pl" in perf_data_daily.columns:
                        total_pl_diff = perf_data_daily["total_pl"].dropna().diff()
                        if not total_pl_diff.empty:
                            total_pl_change_today = total_pl_diff.iloc[-1]
                    if "closed_pl" in perf_data_daily.columns:
                        closed_pl_diff = perf_data_daily["closed_pl"].dropna().diff()
                        if not closed_pl_diff.empty:
                            closed_pl_change_today = closed_pl_diff.iloc[-1]

                    # Monthly Performance
                    monthly_perf_data_last = perf_df.resample("ME").last()
                    monthly_perf_data_first = perf_df.resample("MS").first()

                    if (
                        "total_pl" in monthly_perf_data_last.columns
                        and "total_pl" in monthly_perf_data_first.columns
                        and not monthly_perf_data_last.empty
                        and not monthly_perf_data_first.empty
                    ):
                        # Align indexes before subtracting
                        last_aligned, first_aligned = monthly_perf_data_last[
                            "total_pl"
                        ].align(monthly_perf_data_first["total_pl"], join="inner")
                        if not last_aligned.empty:
                            total_pl_change_MTD = (last_aligned - first_aligned).iloc[
                                -1
                            ]

                    if (
                        "closed_pl" in monthly_perf_data_last.columns
                        and "closed_pl" in monthly_perf_data_first.columns
                        and not monthly_perf_data_last.empty
                        and not monthly_perf_data_first.empty
                    ):
                        # Align indexes before subtracting
                        last_aligned_cl, first_aligned_cl = monthly_perf_data_last[
                            "closed_pl"
                        ].align(monthly_perf_data_first["closed_pl"], join="inner")
                        if not last_aligned_cl.empty:
                            closed_pl_change_MTD = (
                                last_aligned_cl - first_aligned_cl
                            ).iloc[-1]

            except Exception as e_perf:
                logger.error(f"Error calculating performance data: {e_perf}")

        performance_metrics = None
        if hasattr(bot, "calculate_performance_metrics") and callable(
            bot.calculate_performance_metrics
        ):
            performance_metrics = bot.calculate_performance_metrics()

        n_active_orders = len(orders_df)
        n_all_orders = len(all_orders_df)
        n_active_orders_today = 0
        if "DIT" in orders_df.columns:
            n_active_orders_today = len(orders_df[orders_df["DIT"] == 0])

        return {
            "Allocation": allocation,
            "Total P/L": total_pl,
            "Closed P/L": closed_pl,
            "Total P/L Today": total_pl_change_today,
            "Closed P/L Today": closed_pl_change_today,
            "Total P/L MTD": total_pl_change_MTD,
            "Closed P/L MTD": closed_pl_change_MTD,
            "Risk": risk,
            "Available to Trade": available_to_trade,
            "Performance Metrics": performance_metrics,
            "Active Orders": orders_df,
            "All Orders": all_orders_df,
            "Number of Active Orders Today": n_active_orders_today,
            "Number of Active Orders": n_active_orders,
            "Number of Orders": n_all_orders,
        }
    except Exception as e_status:
        logger.exception(f"Error during check_bot_status execution: {e_status}")
        return None  # Indicate failure


def ask_about_today(
    bot_name: str,
    message: str,
    model: str = "qwen2.5-coder:1.5b-instruct",
):
    """
    Summarize today's action by reading the bot's log file and using an
    LLM to summarize the log entries.
    """
    from datetime import datetime

    # Defer ollama import until needed
    # import ollama

    # Construct the log file path based on the bot name
    # Ensure bot_name corresponds to a directory
    log_file_path = f"{bot_name}/{bot_name}.log"
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        logger.error(f"Log file not found at {log_file_path}")
        return

    try:
        with open(log_file_path, "r") as file:
            log_content = file.read()
    except Exception as e_read:
        print(f"Error reading log file {log_file_path}: {e_read}")
        logger.exception(f"Error reading log file {log_file_path}")
        return

    # Extract today's log entries
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    today_entries = [line for line in log_content.splitlines() if today_str in line]

    if not today_entries:
        print("No log entries found for today.")
        logger.info("No log entries found for today.")
        return

    # Join the entries into a single string
    today_entries_str = "\n".join(today_entries)

    try:
        # Import ollama here to avoid making it a hard dependency if not used
        import ollama

        # Set up Ollama client
        client = ollama.Client()  # Consider adding host/port if not default

        # Use Ollama to summarize the log entries
        system_prompt = (
            "You are a helpful assistant. I will ask you questions about a "
            "software log. Your task is to find the accurate answers from "
            "the log ONLY. Do not incorporate any additional information "
            "outside of the context"
        )
        user_prompt = (
            f"Today is {today_str}. The log file reads:\n{today_entries_str}\n\n"
            f"My question for you is: {message}"
        )

        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        reply = response["message"]["content"]
        print(f"Answer:\n{reply}")
        logger.info(f"LLM query successful for bot {bot_name}.")

    except ImportError:
        print("Error: 'ollama' library not installed. Cannot use --ask_today.")
        logger.error("'ollama' library not installed.")
    except Exception as e_ollama:
        print(f"Error communicating with Ollama: {e_ollama}")
        logger.exception("Error communicating with Ollama")


def display_status_cli(status: dict, bot_name: str):
    """Helper function to print status dictionary to console."""
    if status is None:
        print(f"Could not retrieve status for bot: {bot_name}")
        return

    print("-" * 50)
    print(f"Bot Name: {bot_name}")
    print(f"Allocation: {status.get('Allocation', 'N/A')}")
    print(f"Risk: {status.get('Risk', 'N/A')}")
    print(f"Available to Trade: {status.get('Available to Trade', 'N/A')}")
    print(
        "Number of Active Orders Today: "
        f"{status.get('Number of Active Orders Today', 'N/A')}"
    )
    print("Number of Active Orders: " f"{status.get('Number of Active Orders', 'N/A')}")
    print(f"Total number of Orders: {status.get('Number of Orders', 'N/A')}")

    perf_metrics = status.get("Performance Metrics")
    if perf_metrics is not None:
        print("Performance Metrics:\n")
        for metric, value in perf_metrics.items():
            # Format value nicely if it's a number
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:,.2f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print("Performance Metrics: N/A")

    print("\n")
    print("Total P/L (commission included): " f"${status.get('Total P/L', 0):,.2f}")
    print("Closed P/L (commission included): " f"${status.get('Closed P/L', 0):,.2f}")
    print(f"P/L Change Today: ${status.get('Total P/L Today', 0):,.2f}")
    print(f"Closed P/L Change Today: ${status.get('Closed P/L Today', 0):,.2f}")
    print(f"P/L Change MTD: ${status.get('Total P/L MTD', 0):,.2f}")
    print(f"Closed P/L Change MTD: ${status.get('Closed P/L MTD', 0):,.2f}")
    print("=" * 50)

    print("\nAll Orders:\n")
    all_orders_df = status.get("All Orders")
    if all_orders_df is not None and not all_orders_df.empty:
        print(all_orders_df)
    else:
        print("No 'All Orders' data available.")

    print("\n")
    print("\nActive Orders:\n")
    active_orders_df = status.get("Active Orders")
    if active_orders_df is not None and not active_orders_df.empty:
        print(active_orders_df)
    else:
        print("No 'Active Orders' data available.")
    print("\n")


def display_aggregated_status_cli(aggregated_status: dict):
    """Helper function to print aggregated status to console."""
    print("-" * 50)
    print("Aggregated Status (All Bots)")
    print(f"Total Allocation: {aggregated_status.get('Total Allocation', 0):,.2f}")
    print(f"Total Risk: {aggregated_status.get('Total Risk', 0):,.2f}")
    print(
        "Total Available to Trade: "
        f"{aggregated_status.get('Total Available to Trade', 0):,.2f}"
    )
    print(
        "Total P/L (commission included): "
        f"${aggregated_status.get('Total P/L', 0):,.2f}"
    )
    print(
        "Total Closed P/L (commission included): "
        f"${aggregated_status.get('Total Closed P/L', 0):,.2f}"
    )
    print(
        "Total P/L Change Today: "
        f"${aggregated_status.get('Total P/L Today', 0):,.2f}"
    )
    print(
        "Total Closed P/L Change Today: "
        f"${aggregated_status.get('Total Closed P/L Today', 0):,.2f}"
    )
    print("Total P/L Change MTD: " f"${aggregated_status.get('Total P/L MTD', 0):,.2f}")
    print(
        "Total Closed P/L Change MTD: "
        f"${aggregated_status.get('Total Closed P/L MTD', 0):,.2f}"
    )
    print("=" * 50)
    print("\n")

    print("\nAll Orders (Combined):\n")
    all_orders_df = aggregated_status.get("All Orders")
    if all_orders_df is not None and not all_orders_df.empty:
        print(all_orders_df)
    else:
        print("No combined 'All Orders' data available.")

    print("\n")
    print("\nActive Orders (Combined):\n")
    active_orders_df = aggregated_status.get("Active Orders")
    if active_orders_df is not None and not active_orders_df.empty:
        print(active_orders_df)
    else:
        print("No combined 'Active Orders' data available.")


def run_all_bots():
    """Loads, checks status, aggregates, and displays for all bots."""
    logger.info("Checking status of all available bots.")
    available_bots = check_available_bots()
    if not available_bots:
        print("No available bots found.")
        logger.warning("No available bots found.")
        return

    total_allocation = 0
    total_risk = 0
    total_available_to_trade = 0
    total_pl = 0
    total_closed_pl = 0
    total_pl_change = 0
    total_closed_pl_change = 0
    total_pl_change_mtd = 0
    total_closed_pl_change_mtd = 0
    all_active_orders_list = []
    all_orders_list = []

    for bot_name in available_bots:
        logger.info(f"Processing bot: {bot_name}")
        try:
            # Assume pkl file is in a directory named after the bot
            pkl_path = f"{bot_name}/trading_manager{bot_name}.pkl"
            bot = load_bot(pkl_path)
            status = check_bot_status(bot)

            if status:
                total_allocation += status.get("Allocation", 0)
                total_risk += status.get("Risk", 0)
                total_available_to_trade += status.get("Available to Trade", 0)
                total_pl += status.get("Total P/L", 0)
                total_closed_pl += status.get("Closed P/L", 0)
                total_pl_change += status.get("Total P/L Today", 0)
                total_closed_pl_change += status.get("Closed P/L Today", 0)
                total_pl_change_mtd += status.get("Total P/L MTD", 0)
                total_closed_pl_change_mtd += status.get("Closed P/L MTD", 0)

                active_df = status.get("Active Orders")
                if active_df is not None and not active_df.empty:
                    active_df["Bot"] = bot_name  # Add bot identifier
                    all_active_orders_list.append(active_df)

                all_df = status.get("All Orders")
                if all_df is not None and not all_df.empty:
                    all_df["Bot"] = bot_name
                    all_orders_list.append(all_df)

                # Freeze bot state after processing
                # Consider making freeze optional or configurable
                freeze_path = pkl_path  # Freeze back to the same file
                if hasattr(bot, "freeze") and callable(bot.freeze):
                    bot.freeze(freeze_path)
                    logger.info(f"Froze bot state for: {bot_name} to {freeze_path}")
                else:
                    logger.warning(f"Bot {bot_name} has no freeze method.")
            else:
                logger.warning(
                    f"Could not get status for bot {bot_name}. Skipping aggregation."
                )

        except FileNotFoundError:
            logger.error(f"PKL file not found for bot {bot_name}. Skipping.")
        except Exception as e_proc:
            logger.exception(f"Error processing bot {bot_name}: {e_proc}. Skipping.")

    logger.info("Aggregation complete. Preparing display.")

    # Combine DataFrames
    combined_all_orders = pd.DataFrame()
    if all_orders_list:
        combined_all_orders = pd.concat(all_orders_list, ignore_index=True)
        combined_all_orders.sort_values(by="Exit Time", inplace=True)
        combined_all_orders.reset_index(drop=True, inplace=True)

    combined_active_orders = pd.DataFrame()
    if all_active_orders_list:
        combined_active_orders = pd.concat(all_active_orders_list, ignore_index=True)
        # Example sort: by Bot then Description
        if "Description" in combined_active_orders.columns:
            combined_active_orders.sort_values(by=["Bot", "Description"], inplace=True)
        else:
            combined_active_orders.sort_values(by=["Bot"], inplace=True)
        combined_active_orders.reset_index(drop=True, inplace=True)

    aggregated_status = {
        "Total Allocation": total_allocation,
        "Total Risk": total_risk,
        "Total Available to Trade": total_available_to_trade,
        "Total P/L": total_pl,
        "Total Closed P/L": total_closed_pl,
        "Total P/L Today": total_pl_change,
        "Total Closed P/L Today": total_closed_pl_change,
        "Total P/L MTD": total_pl_change_mtd,
        "Total Closed P/L MTD": total_closed_pl_change_mtd,
        "All Orders": combined_all_orders,
        "Active Orders": combined_active_orders,
    }

    display_aggregated_status_cli(aggregated_status)


def run_single_bot(args):
    """Handles logic for operating on a single specified bot."""
    bot_name = args.bot
    logger.info(f"Processing single bot: {bot_name}")

    try:
        # Determine PKL path - assumes bot name is directory name
        # This might need adjustment if CWD isn't the parent of bot dirs
        # Or if pkl isn't named trading_manager<bot_name>.pkl
        pkl_path = f"{bot_name}/trading_manager{bot_name}.pkl"
        if not os.path.exists(pkl_path):
            # Fallback: check if pkl is in CWD (maybe running from bot dir)
            cwd_pkl_path = f"trading_manager{bot_name}.pkl"
            if os.path.exists(cwd_pkl_path):
                pkl_path = cwd_pkl_path
            else:
                raise FileNotFoundError(f"Cannot find PKL for {bot_name}")

        bot = load_bot(pkl_path)

        # --- Bot Actions ---
        action_taken = False
        if args.update_config:
            logger.info(f"Updating configuration for bot: {bot_name}")
            config_updates = {}
            for item in args.update_config:
                if "=" in item:
                    k, v = item.split("=", 1)
                    config_updates[k] = v  # Add type conversion if needed
                else:
                    logger.warning(f"Skipping invalid config item: {item}")
            if config_updates:
                bot.update_config(**config_updates)
                logger.info(f"Configuration updated: {config_updates}")
                action_taken = True

        if args.update_exit:
            logger.info(f"Updating exit scheme for bot: {bot_name}")
            exit_updates = {}
            for item in args.update_exit:
                if "=" in item:
                    k, v = item.split("=", 1)
                    exit_updates[k] = v  # Add type conversion if needed
                else:
                    logger.warning(f"Skipping invalid exit item: {item}")
            if exit_updates:
                update_exit(bot, **exit_updates)
                logger.info(f"Exit scheme updated: {exit_updates}")
                action_taken = True

        if args.liquidate_all:
            logger.info(f"Liquidating all orders for bot: {bot_name}.")
            bot.liquidate_all()
            logger.info("All orders liquidated.")
            print("All orders liquidated.")
            action_taken = True
        elif args.cancel_order:
            order_id = args.cancel_order
            logger.info(f"Canceling order: {order_id} for bot: {bot_name}")
            bot.cancel_order(order_id)
            logger.info(f"Order {order_id} canceled.")
            print(f"Order {order_id} canceled.")
            action_taken = True
        elif args.override_order:
            order_id = args.override_order
            logger.info(f"Overriding order: {order_id} for bot: {bot_name}")
            bot.override_order(order_id)
            logger.info(f"Order {order_id} overridden.")
            print(f"Order {order_id} overridden.")
            action_taken = True
        elif args.close_order:
            order_id = args.close_order
            logger.info(f"Closing order: {order_id} for bot: {bot_name}")
            bot.close_order(order_id)
            logger.info(f"Order {order_id} closed.")
            print(f"Order {order_id} closed.")
            action_taken = True

        # --- Status Check (if no other action taken) ---
        if not action_taken:
            logger.info(f"Checking status of bot: {bot_name}")
            status = check_bot_status(bot)
            display_status_cli(status, bot_name)

        # --- Freeze Bot State ---
        # Consider making freeze optional or configurable, esp. after status check
        freeze_path = pkl_path  # Freeze back to the loaded path
        if hasattr(bot, "freeze") and callable(bot.freeze):
            bot.freeze(freeze_path)
            logger.info(f"Froze bot state for: {bot_name} to {freeze_path}")
        else:
            logger.warning(f"Bot {bot_name} has no freeze method.")

    except FileNotFoundError:
        logger.error(f"PKL file not found for bot {bot_name}.")
        print(f"Error: Could not find trading data for bot '{bot_name}'.")
    except Exception as e_single:
        logger.exception(f"Error processing bot {bot_name}: {e_single}")
        print(f"An error occurred while processing bot '{bot_name}': {e_single}")


def main():
    """Main function to check the status of a trading bot."""
    parser = argparse.ArgumentParser(
        description="Check status or perform actions on trading bots."
    )
    parser.add_argument(
        "bot",
        help="The bot name to check, or 'all' for aggregated status.",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--list_bots", action="store_true", help="List available bot names."
    )
    # Actions (Mutually exclusive with status check for 'all')
    parser.add_argument("--close_order", help="Order ID to close (requires bot name).")
    parser.add_argument(
        "--cancel_order", help="Order ID to cancel (requires bot name)."
    )
    parser.add_argument(
        "--override_order", help="Order ID to override (requires bot name)."
    )
    parser.add_argument(
        "--liquidate_all",
        action="store_true",
        help="Liquidate all orders for the specified bot (requires bot name).",
    )
    parser.add_argument(
        "--update_config",
        nargs="*",
        metavar="KEY=VALUE",
        help="Update bot configuration (KEY=VALUE pairs, requires bot name).",
    )
    parser.add_argument(
        "--update_exit",
        nargs="*",
        metavar="KEY=VALUE",
        help="Update exit scheme (KEY=VALUE pairs, requires bot name).",
    )
    # LLM Query
    parser.add_argument(
        "--ask_today",
        help="Ask a question about today's log entries for the specified bot.",
    )
    parser.add_argument(
        "--model",
        help="LLM model for --ask_today",
        default="qwen2.5-coder:1.5b-instruct",
    )
    # Removed api_key and base_url as ollama client likely handles config

    args = parser.parse_args()

    # --- Mode Dispatch ---
    if args.list_bots:
        if args.bot is not None:
            parser.error("--list_bots does not take a bot name argument.")
        logger.info("Listing available bots.")
        available_bots = check_available_bots()
        print("Available Bots:\n")
        if available_bots:
            for bot_name in sorted(list(available_bots)):
                print(bot_name)
        else:
            print("No bots found.")

    elif args.ask_today:
        if not args.bot or args.bot == "all":
            parser.error("--ask_today requires a specific bot name.")
        # Assume bot name corresponds to directory name for logs
        ask_about_today(args.bot, args.ask_today, model=args.model)

    elif args.bot == "all":
        # Check if any single-bot actions were specified with 'all'
        action_args = [
            args.close_order,
            args.cancel_order,
            args.override_order,
            args.liquidate_all,
            args.update_config,
            args.update_exit,
        ]
        if any(action_args):
            parser.error(
                "Actions like --close_order, --liquidate_all, etc., "
                "cannot be used with bot name 'all'."
            )
        run_all_bots()

    elif args.bot:
        # Handle single bot status check or actions
        run_single_bot(args)

    else:
        # No bot specified, no --list_bots -> show help
        parser.print_help()

    logger.info("bot_status script finished.\n")


if __name__ == "__main__":
    main()
