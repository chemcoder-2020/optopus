import streamlit as st
import pandas as pd
from loguru import logger


# Placeholder function for displaying bot status
# It receives the selected bot name and the necessary functions from bot_status
def display_status(bot_name, load_func, status_func, available_func):
    """
    Displays the status information for the selected bot or all bots.

    Args:
        bot_name (str): The name of the selected bot ('all' for aggregate).
        load_func (callable): The function to load a bot (e.g., load_bot).
        status_func (callable): Function to check bot status
            (e.g., check_bot_status).
        available_func (callable): Function to get available bots
            (e.g., check_available_bots).
    """
    # # Removed placeholder write
    # st.write(f"Displaying status for: {bot_name}")
    # Removed placeholder info
    # st.info("Status display implementation is pending.")

    if bot_name == "all":
        display_all_bots_status(status_func, available_func, load_func)
    else:
        display_single_bot_status(bot_name, load_func, status_func)


def display_single_bot_status(bot_name, load_func, status_func):
    """Displays status for a single selected bot."""
    try:
        # Construct the expected path for the pkl file
        # Assuming bots are in directories named after them at the root level
        # This might need adjustment based on actual project structure
        pkl_path = f"{bot_name}/trading_manager{bot_name}.pkl"
        bot = load_func(pkl_path)
        status = status_func(bot)

        # Display Performance Chart if available
        if hasattr(bot, 'performance_data') and bot.performance_data:
            st.subheader("Performance Chart")
            perf_df = pd.DataFrame(bot.performance_data)
            fig = bot._plot_interactive_performance(perf_df)
            st.plotly_chart(fig, use_container_width=True)

        # Display Key Metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Allocation", f"${status.get('Allocation', 0):,.2f}")
        col2.metric("Risk", f"${status.get('Risk', 0):,.2f}")
        col3.metric(
            "Available to Trade",
            f"${status.get('Available to Trade', 0):,.2f}",
        )

        col1a, col2a, col3a = st.columns(3)
        col1a.metric("Total P/L", f"${status.get('Total P/L', 0):,.2f}")
        col2a.metric("Closed P/L", f"${status.get('Closed P/L', 0):,.2f}")
        # Placeholder for Open P/L if needed: Total P/L - Closed P/L
        open_pl = status.get("Total P/L", 0) - status.get("Closed P/L", 0)
        col3a.metric("Open P/L", f"${open_pl:,.2f}")

        col1b, col2b = st.columns(2)
        col1b.metric(
            "Total P/L Today", f"${status.get('Total P/L Today', 0):,.2f}"
        )
        col2b.metric(
            "Closed P/L Today",
            f"${status.get('Closed P/L Today', 0):,.2f}",
        )

        col1c, col2c = st.columns(2)
        col1c.metric(
            "Total P/L MTD", f"${status.get('Total P/L MTD', 0):,.2f}"
        )
        col2c.metric(
            "Closed P/L MTD",
            f"${status.get('Closed P/L MTD', 0):,.2f}",
        )

        # Display Active Orders
        st.subheader("Active Orders")
        active_orders_df = status.get("Active Orders", pd.DataFrame())
        if not active_orders_df.empty:
            st.dataframe(active_orders_df)
        else:
            st.write("No active orders.")

        # Display All Orders (Optional - can be large)
        # st.subheader("All Orders History")
        # all_orders_df = status.get("All Orders", pd.DataFrame())
        # if not all_orders_df.empty:
        #     st.dataframe(all_orders_df)
        # else:
        #     st.write("No order history.")

        # Display Performance Metrics (if available)
        perf_metrics = status.get("Performance Metrics")
        if perf_metrics:
            st.subheader("Performance Metrics")
            st.json(perf_metrics)  # Simple display for now

    except FileNotFoundError:
        st.error(
            f"Could not find trading_manager pkl file for bot: {bot_name} at "
            f"expected path: {pkl_path}",
        )
    except Exception as e:
        logger.exception(
            f"Error loading or checking status for bot {bot_name}"
        )
        st.error(f"Processing error for bot {bot_name}: {e}")


def display_all_bots_status(status_func, available_func, load_func):
    """Displays aggregated status for all available bots."""
    try:
        available_bots = available_func()
        if not available_bots:
            st.warning("No bots found.")
            return

        total_allocation = 0
        total_risk = 0
        total_available_to_trade = 0
        total_pl = 0
        total_closed_pl = 0
        total_pl_change_today = 0
        total_closed_pl_change_today = 0
        total_pl_change_mtd = 0
        total_closed_pl_change_mtd = 0
        all_active_orders_list = []
        # all_orders_list = []  # Keep commented out unless needed

        progress_bar = st.progress(0)
        total_bots = len(available_bots)

        for i, bot_name in enumerate(available_bots):
            try:
                pkl_path = f"{bot_name}/trading_manager{bot_name}.pkl"
                bot = load_func(pkl_path)
                status = status_func(bot)

                total_allocation += status.get("Allocation", 0)
                total_risk += status.get("Risk", 0)
                total_available_to_trade += status.get(
                    "Available to Trade", 0
                )
                total_pl += status.get("Total P/L", 0)
                total_closed_pl += status.get("Closed P/L", 0)
                total_pl_change_today += status.get("Total P/L Today", 0)
                total_closed_pl_change_today += status.get(
                    "Closed P/L Today", 0
                )
                total_pl_change_mtd += status.get("Total P/L MTD", 0)
                total_closed_pl_change_mtd += status.get(
                    "Closed P/L MTD", 0
                )

                active_df = status.get("Active Orders", pd.DataFrame())
                if not active_df.empty:
                    active_df["Bot"] = bot_name  # Add bot identifier
                    all_active_orders_list.append(active_df)

                # all_df = status.get("All Orders", pd.DataFrame())
                # if not all_df.empty:
                #     all_df['Bot'] = bot_name
                #     all_orders_list.append(all_df)

            except FileNotFoundError:
                st.warning(
                    f"Could not find pkl file for bot: {bot_name} at "
                    f"expected path: {pkl_path}. Skipping."
                )
            except Exception as e:
                logger.warning(
                    f"Could not process bot {bot_name}: {e}. Skipping."
                )
            finally:
                progress_bar.progress((i + 1) / total_bots)

        # Display Aggregated Metrics
        st.subheader("Aggregated Metrics (All Bots)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Allocation", f"${total_allocation:,.2f}")
        col2.metric("Total Risk", f"${total_risk:,.2f}")
        col3.metric(
            "Total Available",
            f"${total_available_to_trade:,.2f}",
        )

        col1a, col2a, col3a = st.columns(3)
        col1a.metric("Total P/L", f"${total_pl:,.2f}")
        col2a.metric("Total Closed P/L", f"${total_closed_pl:,.2f}")
        total_open_pl = total_pl - total_closed_pl
        col3a.metric("Total Open P/L", f"${total_open_pl:,.2f}")

        col1b, col2b = st.columns(2)
        col1b.metric("Total P/L Today", f"${total_pl_change_today:,.2f}")
        col2b.metric(
            "Total Closed P/L Today",
            f"${total_closed_pl_change_today:,.2f}",
        )

        col1c, col2c = st.columns(2)
        col1c.metric("Total P/L MTD", f"${total_pl_change_mtd:,.2f}")
        col2c.metric(
            "Total Closed P/L MTD",
            f"${total_closed_pl_change_mtd:,.2f}",
        )

        # Display Combined Active Orders
        st.subheader("All Active Orders")
        if all_active_orders_list:
            combined_active_df = pd.concat(
                all_active_orders_list, ignore_index=True
            )
            # Sort or rearrange columns if needed, e.g., put 'Bot' first
            cols = ["Bot"] + [
                col for col in combined_active_df.columns if col != "Bot"
            ]
            st.dataframe(combined_active_df[cols])
        else:
            st.write("No active orders across all bots.")

        # Display Combined All Orders (Optional)
        # st.subheader("All Order History")
        # if all_orders_list:
        #     combined_all_df = pd.concat(all_orders_list, ignore_index=True)
        #     cols = ['Bot'] +
        # [col for col in combined_all_df.columns if col != 'Bot']
        #     st.dataframe(combined_all_df[cols])
        # else:
        #     st.write("No order history across all bots.")

    except Exception as e:
        logger.exception("Error displaying aggregated bot status")
        st.error(f"An error occurred during aggregation: {e}")
