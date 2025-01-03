import os
import dill
import argparse
import pandas as pd
from loguru import logger
import glob
from pathlib import Path

logger.enable("optopus")

# Set up paths relative to the package
package_dir = Path(__file__).parent.parent.parent
os.chdir(package_dir)

logger.add("bot_status.log", rotation="10 MB", retention="60 days", compression="zip")

pd.options.display.max_columns = 50

def check_available_bots():
    """Recursively check for any pkl files and report the name of their immediate parent folder."""
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    available_bots = set(os.path.basename(os.path.dirname(pkl)) for pkl in pkl_files)
    return available_bots

# ... [rest of the existing bot_status.py code] ...

if __name__ == "__main__":
    main()
