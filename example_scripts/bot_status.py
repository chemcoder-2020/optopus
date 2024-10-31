import os
import pickle
import argparse
import pandas as pd
from loguru import logger
import glob

logger.enable("optopus")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
basename = os.path.split(dname)[0]

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
        pkl_path (str): Path to the pickle file.

    Returns:
        The loaded bot instance.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at: {pkl_path}")
    with open(pkl_path, "rb") as file:
        bot = pickle.load(file)
    return bot


def check_bot_status(bot):
    """Check the status of a bot.

    Args:
        bot: The bot instance to check.

    Returns:
        A dictionary containing the bot's status.
    """
    bot.auth_refresh()
    bot.update_orders()
    orders_df = bot.get_active_orders_dataframe()
    all_orders_df = bot.get_orders_dataframe()
    allocation = bot.allocation
    risk = sum(trade.get_required_capital() for trade in bot.active_orders)
    total_pl = all_orders_df["Total P/L"].sum()
    closed_pl = all_orders_df[all_orders_df["Status"] == "CLOSED"]["Total P/L"].sum()
    performance_metrics = bot.calculate_performance_metrics()
    return {
        "Allocation": allocation,
        "Total P/L": total_pl,
        "Closed P/L": closed_pl,
        "Risk": risk,
        "Performance Metrics": performance_metrics,
        "Active Orders": orders_df,
        "All Orders": all_orders_df,
    }


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
    args = parser.parse_args()

    if args.list_bots and args.bot is not None:
        parser.error("The --list_bots option does not require a bot argument.")

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
        total_pl = 0
        total_closed_pl = 0
        all_active_orders = []
        all_orders = []

        for bot_name in available_bots:
            name = bot_name.split("Bot")[0]
            logger.info(f"Loading bot: {name}")
            bot = load_bot(f"{name}Bot/trading_manager{name}.pkl")
            status = check_bot_status(bot)
            bot.freeze(f"{name}Bot/trading_manager{name}.pkl")
            logger.info(f"Froze bot: {name}")
            total_allocation += status["Allocation"]
            total_risk += status["Risk"]
            total_pl += status["Total P/L"]
            total_closed_pl += status["Closed P/L"]
            all_active_orders.append(status["Active Orders"])
            all_orders.append(status["All Orders"])

        logger.info("Aggregating and printing bot status.")
        print("-" * 50)
        print(f"Total Allocation: {total_allocation}")
        print(f"Total Risk: {total_risk}")
        print(f"Total P/L (commission included): {total_pl:.2f}")
        print(f"Total Closed P/L (commission included): {total_closed_pl:.2f}")
        print("=" * 50)
        print("\nActive Orders:\n")
        print(
            pd.concat(
                [df for df in all_active_orders if not df.empty], ignore_index=True
            )
            .sort_values(by="Description")
            .reset_index(drop=True)
        )
        print("\n")

        print("\nAll Orders:\n")
        print(
            pd.concat([df for df in all_orders if not df.empty], ignore_index=True)
            .sort_values(by="Exit Time")
            .reset_index(drop=True)
        )
        print("\n")

    else:
        logger.info(f"Loading bot: {args.bot}")
        bot = load_bot(f"{args.bot}Bot/trading_manager{args.bot}.pkl")

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
            logger.info(f"Performance Metrics: {status['Performance Metrics']}")
            logger.info(f"Total P/L (commission included): {status['Total P/L']:.2f}")
            logger.info(f"Closed P/L (commission included): {status['Closed P/L']:.2f}")
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
            if status['Performance Metrics'] is not None:
                print(f"Performance Metrics:\n")
                for metric, value in status["Performance Metrics"].items():
                    print(f"  {metric}: {value}")

            print("\n")
            print(f"Total P/L (commission included): {status['Total P/L']:.2f}")
            print(f"Closed P/L (commission included): {status['Closed P/L']:.2f}")
            print("=" * 50)
            print("\nActive Orders:\n")
            print(status["Active Orders"])
            print("\n")
            print("\nAll Orders:\n")
            print(status["All Orders"])
            print("\n")
        bot.freeze(f"{args.bot}Bot/trading_manager{args.bot}.pkl")
        logger.info(f"Froze bot: {args.bot}")

    logger.info("Done.\n\n")


if __name__ == "__main__":
    main()
