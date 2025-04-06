# import streamlit.cli # Don't import internal CLI
import os
import sys
import subprocess


def main():
    """
    Finds the app.py file relative to this runner script and executes
    streamlit run on it.
    """
    # Get the directory where this runner script is located
    runner_dir = os.path.dirname(__file__)
    # Construct the absolute path to app.py
    app_path = os.path.join(runner_dir, "app.py")

    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Prepare arguments for streamlit run
    # We include the --server.fileWatcherType none flag as it was needed before
    # Prepare arguments for streamlit run command using subprocess
    command = [
        sys.executable,  # Use the current python interpreter
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.fileWatcherType",
        "none",
    ]

    print(f"Running command: {' '.join(command)}")
    # Execute the command
    subprocess.run(command)


if __name__ == "__main__":
    main()
