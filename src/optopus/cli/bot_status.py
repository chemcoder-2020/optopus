import os
import sys
import dill
import argparse
import pandas as pd
from loguru import logger
import glob
from pathlib import Path

# Set up logging
logger.add("bot_status.log", rotation="10 MB", retention="60 days", compression="zip")

pd.options.display.max_columns = 50


def check_available_bots():
    """Recursively check for any pkl files and report the name of their immediate parent folder."""
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    available_bots = set(os.path.basename(os.path.dirname(pkl)) for pkl in pkl_files)
    return available_bots


def load_bot(pkl_path: str):
    """Load a bot from a pickle file.

    Args:
        pkl_path (str): Path to the pickle file or directory containing trading_manager.pkl.

    Returns:
        The loaded bot instance.
    """

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {pkl_path}")

    # Get the directory containing the pickle file
    pkl_dir = os.path.dirname(os.path.abspath(pkl_path))

    # Store current directory and sys.path
    original_dir = os.getcwd()
    original_sys_path = sys.path.copy()

    try:
        # Change to the pickle file's directory
        os.chdir(pkl_dir)

        # Add the directory to Python path
        if pkl_dir not in sys.path:
            sys.path.insert(0, pkl_dir)

        # Load the bot
        with open(os.path.basename(pkl_path), "rb") as file:
            bot = dill.load(file)

        return bot
    finally:
        # Always change back to original directory and restore sys.path
        os.chdir(original_dir)
        sys.path = original_sys_path


def update_exit(bot, **kwargs):
    """Update the exit scheme for each active order.

    Args:
        bot: The bot instance to update.
        **kwargs: Keyword arguments to pass to the exit scheme update method.
    """
    for order in bot.active_orders:
        if hasattr(order, "exit_scheme") and callable(
            getattr(order.exit_scheme, "update", None)
        ):
            order.exit_scheme.update(**kwargs)


def check_bot_status(bot):
    """Check the status of a bot.

    Args:
        bot: The bot instance to check.

    Returns:
        A dictionary containing the bot's status.
    """
    try:
        bot.update_orders()
    except Exception as e:
        bot.auth_refresh()
        bot.update_orders()

    orders_df = bot.get_active_orders_dataframe()
    orders_df.sort_values(by="Entry Time", inplace=True)
    all_orders_df = bot.get_orders_dataframe()
    all_orders_df.sort_values(by="Exit Time", inplace=True)
    allocation = bot.allocation
    risk = sum(trade.get_required_capital() for trade in bot.active_orders)
    available_to_trade = bot.available_to_trade
    total_pl = all_orders_df["Total P/L"].sum()
    closed_pl = all_orders_df[all_orders_df["Status"] == "CLOSED"]["Total P/L"].sum()

    # Daily performance
    perf_data = (
        pd.DataFrame(bot.performance_data).set_index("time").resample("D").last()
    )
    total_pl_change_today = perf_data["total_pl"].dropna().diff().iloc[-1]
    closed_pl_change_today = perf_data["closed_pl"].dropna().diff().iloc[-1]

    # Monthly Performance
    monthly_perf_data_last = (
        pd.DataFrame(bot.performance_data).set_index("time").resample("ME").last()
    )
    monthly_perf_data_first = (
        pd.DataFrame(bot.performance_data).set_index("time").resample("MS").first()
    )
    total_pl_change_MTD = (
        monthly_perf_data_last["total_pl"].reset_index(drop=True)
        - monthly_perf_data_first["total_pl"].reset_index(drop=True)
    ).iloc[-1]
    closed_pl_change_MTD = (
        monthly_perf_data_last["closed_pl"].reset_index(drop=True)
        - monthly_perf_data_first["closed_pl"].reset_index(drop=True)
    ).iloc[-1]

    performance_metrics = bot.calculate_performance_metrics()
    n_active_orders = len(orders_df)
    n_all_orders = len(all_orders_df)
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


def ask_about_today(
    bot_name: str,
    message: str,
    model: str = "qwen2.5-coder:1.5b-instruct",
):
    """Summarize today's action by reading the bot's log file and using an LLM to summarize the log entries."""
    from datetime import datetime
    import ollama

    # Construct the log file path based on the bot name
    log_file_path = f"{bot_name}/{bot_name}.log"
    with open(log_file_path, "r") as file:
        log_content = file.read()

    # # Extract today's log entries
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    today_entries = [line for line in log_content.splitlines() if today_str in line]

    # # Join the entries into a single string
    today_entries_str = "\n".join(today_entries)

    # Set up Ollama client
    client = ollama.Client()

    # Use Ollama to summarize the log entries

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. I will ask you questions about a software log. Your task is to find the accurate answers from the log ONLY. Do not incorporate any additional information outside of the context",
            },
            {
                "role": "user",
                "content": f"Today is {today_str}. The log file reads:\n{today_entries_str}\n\n My question for you is: {message}",
            },
        ],
    )

    reply = response["message"]["content"]
    print(f"Answer:\n{reply}")


def main():
    """Main function to check the status of a trading bot.

    Parses command-line arguments and prints the bot's status.
    """
    parser = argparse.ArgumentParser(description="Check the status of trading bots.")
    parser.add_argument("bot", help="The bot to check", nargs="?", default=None)
    parser.add_argument("--close_order", help="Order ID to close")
    parser.add_argument("--cancel_order", help="Order ID to cancel")
    parser.add_argument("--override_order", help="Order ID to override")
    parser.add_argument(
        "--liquidate_all", action="store_true", help="Liquidate all orders"
    )
    parser.add_argument("--list_bots", action="store_true", help="List available bots")
    parser.add_argument("--ask_today", help="Ask about today")
    parser.add_argument("--api_key", help="API key for the LLM")
    parser.add_argument("--base_url", help="Base URL for the LLM")
    parser.add_argument(
        "--model",
        help="Model to use for the LLM",
        default="qwen2.5-coder:1.5b-instruct",
    )
    parser.add_argument(
        "--update_config",
        nargs="*",
        metavar="KEY=VALUE",
        help="Update bot configuration",
    )
    parser.add_argument(
        "--update_exit",
        nargs="*",
        metavar="KEY=VALUE",
        help="Update exit scheme for active orders",
    )
    args = parser.parse_args()

    if args.list_bots and args.bot is not None:
        parser.error("The --list_bots option does not require a bot argument.")

    if args.ask_today:
        if not args.bot:
            parser.error(
                "The --summarize_today option requires --api_key, --base_url, and a bot name."
            )
        ask_about_today(args.bot + "Bot", args.ask_today, model=args.model)
        return

    if args.list_bots:
        logger.info("Listing available bots.")
        available_bots = check_available_bots()
        print("Available Bots:\n")
        for bot in available_bots:
            print(bot)
        return

    if args.bot == "all":
        logger.info("Checking status of all available bots.")
        available_bots = check_available_bots()
        total_allocation = 0
        total_risk = 0
        total_available_to_trade = 0
        total_pl = 0
        total_closed_pl = 0
        total_pl_change = 0
        total_closed_pl_change = 0
        total_pl_change_mtd = 0
        total_closed_pl_change_mtd = 0
        all_active_orders = []
        all_orders = []

        for bot_name in available_bots:
            name = bot_name
            logger.info(f"Loading bot: {name}")
            bot = load_bot(f"{name}/trading_manager{name}.pkl")
            status = check_bot_status(bot)
            bot.freeze(f"{name}/trading_manager{name}.pkl")
            logger.info(f"Froze bot: {name}")
            total_allocation += status["Allocation"]
            total_risk += status["Risk"]
            total_available_to_trade += status["Available to Trade"]
            total_pl += status["Total P/L"]
            total_closed_pl += status["Closed P/L"]
            total_pl_change += status["Total P/L Today"]
            total_closed_pl_change += status["Closed P/L Today"]
            total_pl_change_mtd += status["Total P/L MTD"]
            total_closed_pl_change_mtd += status["Closed P/L MTD"]
            all_active_orders.append(status["Active Orders"])
            all_orders.append(status["All Orders"])

        logger.info("Aggregating and printing bot status.")
        print("-" * 50)
        print(f"Total Allocation: {total_allocation}")
        print(f"Total Risk: {total_risk}")
        print(f"Total Available to Trade: {total_available_to_trade}")
        print(f"Total P/L (commission included): ${total_pl:.2f}")
        print(f"Total Closed P/L (commission included): ${total_closed_pl:.2f}")
        print(f"Total P/L Change Today: ${total_pl_change:.2f}")
        print(f"Total Closed P/L Change Today: ${total_closed_pl_change:.2f}")
        print(f"Total P/L Change MTD: ${total_pl_change_mtd:.2f}")
        print(f"Total Closed P/L Change MTD: ${total_closed_pl_change_mtd:.2f}")
        print("=" * 50)

        print("\n")

        print("\nAll Orders:\n")
        print(
            pd.concat([df for df in all_orders if not df.empty], ignore_index=True)
            .sort_values(by="Exit Time")
            .reset_index(drop=True)
        )
        print("\n")
        print("\nActive Orders:\n")
        print(
            pd.concat(
                [df for df in all_active_orders if not df.empty], ignore_index=True
            )
            .sort_values(by="Description")
            .reset_index(drop=True)
        )

    else:
        logger.info(f"Loading bot: {args.bot}")
        original_dir = os.getcwd()
        if args.bot == os.path.split(original_dir)[-1]:
            bot = load_bot(f"trading_manager{args.bot}.pkl")
        else:
            bot = load_bot(f"{args.bot}/trading_manager{args.bot}.pkl")

        if args.update_config:
            logger.info(f"Updating configuration for bot: {args.bot}")
            config_updates = {
                k: v for k, v in (item.split("=") for item in args.update_config)
            }
            bot.update_config(**config_updates)
            logger.info(f"Configuration updated: {config_updates}")

        if args.update_exit:
            logger.info(f"Updating exit scheme for bot: {args.bot}")
            exit_updates = {
                k: v for k, v in (item.split("=") for item in args.update_exit)
            }
            update_exit(bot, **exit_updates)
            logger.info(f"Exit scheme updated: {exit_updates}")

        if args.liquidate_all:
            logger.info("Liquidating all orders.")
            bot.liquidate_all()
            logger.info("All orders liquidated.")
            print("All orders liquidated.")
        elif args.cancel_order:
            logger.info(f"Canceling order: {args.cancel_order}")
            bot.cancel_order(args.cancel_order)
            logger.info(f"Order {args.cancel_order} canceled.")
            print(f"Order {args.cancel_order} canceled.")
        elif args.override_order:
            logger.info(f"Overriding order: {args.override_order}")
            bot.override_order(args.override_order)
            logger.info(f"Order {args.override_order} overridden.")
            print(f"Order {args.override_order} overridden.")
        elif args.close_order:
            logger.info(f"Closing order: {args.close_order}")
            bot.close_order(args.close_order)
            logger.info(f"Order {args.close_order} closed.")
            print(f"Order {args.close_order} closed.")
        else:
            logger.info(f"Checking status of bot: {args.bot}")
            status = check_bot_status(bot)
            logger.info(f"Bot Name: {args.bot}")
            logger.info(f"Allocation: {status['Allocation']}")
            logger.info(f"Risk: {status['Risk']}")
            logger.info(f"Available to Trade: {status['Available to Trade']}")
            logger.info(
                f"Number of Active Orders Today: {status['Number of Active Orders Today']}"
            )
            logger.info(f"Number of Active Orders: {status['Number of Active Orders']}")
            logger.info(f"Total number of Orders: {status['Number of Orders']}")
            logger.info(f"Performance Metrics: {status['Performance Metrics']}")
            logger.info(f"Total P/L (commission included): ${status['Total P/L']:.2f}")
            logger.info(
                f"Closed P/L (commission included): ${status['Closed P/L']:.2f}"
            )
            logger.info(f"P/L Change Today: ${status["Total P/L Today"]:.2f}")
            logger.info(f"Closed P/L Change Today: ${status["Closed P/L Today"]:.2f}")
            logger.info(f"P/L Change MTD: ${status["Total P/L MTD"]:.2f}")
            logger.info(f"Closed P/L Change MTD: ${status["Closed P/L MTD"]:.2f}")

            logger.info("=" * 50)
            logger.info("\nActive Orders:\n")
            logger.info(status["Active Orders"])
            logger.info("\n")
            logger.info("\nAll Orders:\n")
            logger.info(status["All Orders"])

            print("-" * 50)
            print(f"Bot Name: {args.bot}")
            print(f"Allocation: {status['Allocation']}")
            print(f"Risk: {status['Risk']}")
            print(f"Available to Trade: {status['Available to Trade']}")
            print(
                f"Number of Active Orders Today: {status['Number of Active Orders Today']}"
            )
            print(f"Number of Active Orders: {status['Number of Active Orders']}")
            print(f"Total number of Orders: {status['Number of Orders']}")
            if status["Performance Metrics"] is not None:
                print(f"Performance Metrics:\n")
                for metric, value in status["Performance Metrics"].items():
                    print(f"  {metric}: {value}")

            print("\n")
            print(f"Total P/L (commission included): ${status['Total P/L']:.2f}")
            print(f"Closed P/L (commission included): ${status['Closed P/L']:.2f}")
            print(f"P/L Change Today: ${status["Total P/L Today"]:.2f}")
            print(f"Closed P/L Change Today: ${status["Closed P/L Today"]:.2f}")
            print(f"P/L Change MTD: ${status["Total P/L MTD"]:.2f}")
            print(f"Closed P/L Change MTD: ${status["Closed P/L MTD"]:.2f}")
            print("=" * 50)
            print("\nAll Orders:\n")
            print(status["All Orders"])
            print("\n")
            print("\nActive Orders:\n")
            print(status["Active Orders"])
            print("\n")

        if args.bot == os.path.split(original_dir)[-1]:
            bot.freeze(f"trading_manager{args.bot}.pkl")
        else:
            bot.freeze(f"{args.bot}/trading_manager{args.bot}.pkl")
        logger.info(f"Froze bot: {args.bot}")

    logger.info("Done.\n\n")


if __name__ == "__main__":
    main()
