import streamlit as st

# import os # Unused
# import sys # Unused
from loguru import logger

# Configure the Streamlit page FIRST
st.set_page_config(page_title="Optopus Bot Dashboard", layout="wide")

# # Add the project root to the Python path - REMOVED (not needed for installed package)
# project_root = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..")
# )
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


# Attempt to import functions from bot_status using package path
try:
    # Use package path for installed package
    from optopus.cli.bot_status import (
        check_available_bots,
        load_bot,
        check_bot_status,
    )
except ImportError as e:
    st.error(f"Failed to import functions from bot_status: {e}")
    st.stop()  # Stop execution if imports fail


# Import the display component using absolute path
try:
    from optopus.web.components.status_display import display_status
except ImportError as e:
    st.error(f"Failed to import display_status component: {e}")
    # Create a dummy function if import fails to avoid crashing later

    def display_status(
        bot_name, load_func, status_func, available_func
    ):  # Match signature
        st.warning("Status display component not loaded correctly.")


# Configure the Streamlit page - MOVED TO TOP

st.title("🐙 Optopus Bot Dashboard")

# --- Bot Selection ---
try:
    available_bots = list(check_available_bots())
    available_bots.sort()
    bot_options = ["all"] + available_bots
except Exception as e:
    st.error(f"Error getting available bots: {e}")
    available_bots = []
    bot_options = ["all"]

selected_bot = st.sidebar.selectbox(
    "Select Bot",
    options=bot_options,
    index=0,  # Default to 'all'
)

# --- Display Status ---
if selected_bot:
    st.header(f"Status for: {selected_bot}")
    try:
        # Pass necessary functions to the display component
        # This avoids loading the bot twice (once here, once in the component)
        # We'll refine this interaction later
        display_status(selected_bot, load_bot, check_bot_status, check_available_bots)
    except Exception as e:
        logger.exception(f"Error displaying status for {selected_bot}")
        st.error(f"An error occurred displaying status for {selected_bot}: {e}")
else:
    st.info("Select a bot from the sidebar to view its status.")
