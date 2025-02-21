import argparse
import os
import shutil


def create_directory(project_name):
    """Creates the project directory."""
    os.makedirs(project_name, exist_ok=True)


def copy_template_files(project_name, strategy):
    """Copies template files to the project directory."""
    templates_base_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    common_template_dir = os.path.join(templates_base_dir, "common")
    strategy_template_dir = os.path.join(templates_base_dir, strategy)
    common_files_to_copy = [
        "backtest.py",
        "backtest_cross_validate.py",
        "bot.py",
        "entry_condition.py",
        "external_entry_condition.py",
        "exit_condition.py",
    ]
    strategy_files_to_copy = [
        "strategy_selection.py",
        "config.ini",
    ]

    for file_name in common_files_to_copy:
        source_path = os.path.join(common_template_dir, file_name)
        dest_path = os.path.join(project_name, file_name)

        # Check if the template file exists
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Template file not found: {source_path}. "
                f"Please ensure that the file exists for the common template."
            )

        shutil.copy2(source_path, dest_path)

    for file_name in strategy_files_to_copy:
        source_path = os.path.join(strategy_template_dir, file_name)
        dest_path = os.path.join(project_name, file_name)

        # Check if the template file exists
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Template file not found: {source_path}. "
                f"Please ensure that the file exists for the strategy '{strategy}'."
            )

        shutil.copy2(source_path, dest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Set up a backtesting project for options trading.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "project_name",
        help="Name of the backtesting project directory. This will be the name of the new folder created for your project.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "NakedCall",
            "NakedPut",
            "VerticalSpread",
            "IronCondor",
            "Straddle",
            "IronButterfly",
        ],
        default="VerticalSpread",
        help="The strategy to use for the backtest. Currently supports: NakedCall, NakedPut, VerticalSpread, IronCondor, Straddle, IronButterfly.",
    )

    args = parser.parse_args()

    create_directory(args.project_name)
    copy_template_files(args.project_name, args.strategy)

    print(f"Backtesting project '{args.project_name}' created successfully!")


if __name__ == "__main__":
    main()
