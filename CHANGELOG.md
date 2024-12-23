# Changelog

## v0.9.1-dev5

### Features

-   Added command-line tool `setup-optopus-backtest` for creating backtesting projects.
-   Implemented `create_directory`, `generate_config_file`, `copy_template_files`, and `copy_optopus_files` functions in `setup_backtest.py`.
-   Added template files for `backtest.py`, `config.py`, `entry_condition.py`, `exit_condition.py`, and `backtest_cross_validate.py` in `templates/0dte-IB/`.
-   Included `setup-optopus-backtest` in package installation using `entry_points` in `setup.py`.
-   Added `argparse` to `setup_backtest.py` to handle command-line arguments for project configuration.
-   Added help descriptions for each command-line argument in `setup_backtest.py`.

### Enhancements

-   Updated `generate_config_file` to use `configparser` and allow users to specify values for `config` and `strategy_params` via the command line.
-   Updated `setup.py` and `pyproject.toml` to version `0.9.1-dev5`.
-   Updated `config.py` template to use `CompositeEntryCondition` and remove `MedianCalculator` from entry conditions.
-   Updated `exit_condition.py` template to use `CompositeExitCondition`.

### Refactors

-   Reorganized project structure to include `src/optopus/cli/`.
-   Renamed files to use forward slashes instead of backslashes.

## v0.9.1-dev4

### Performance

-   Improved performance of option chain processing

## v0.9.1-dev3

### Features

-   Added strategies to `__init__` and update `__all__` list
-   Added new modules to `__init__.py`
-   Added `__all__` to strategies `__init__.py`
-   Imported strategy classes in strategies init file
-   Created init file for strategies module
-   Added `VerticalSpread` class and `create_iron_condor` method
-   Moved strategy implementations to dedicated modules

### Documentation

-   Add column alias handling to `validate_option_data` to changelog

## v0.9.1-dev2

### Features

-   Added `MedianCalculator` class to calculate the median of a rolling window of premiums.
-   Updated `ProfitTargetCondition`, `StopLossCondition`, and `TrailingStopCondition` classes to use the `MedianCalculator` for exit decisions.
-   Updated `DefaultExitCondition` class to use the `ProfitTargetCondition` and `TimeBasedCondition` classes to create a composite exit condition.

## v0.9.1-dev1

### Features

-   Update version to `0.9.1-dev1` and add new option data columns
-   Add `exit_dte` to `calculate_performance_metrics` and include average `exit_dte` in metrics
-   Add scenarios to option data validator examples
-   Add example usage of option data validator
-   Handle optional columns in option data validator
-   Add option data validator with schema and type checks
-   Add option data validator utility
-   Add open interest to `OptionLeg` class
-   Add `entry_dte` to trade history
-   Add `entry_dte` to `OptionStrategy` creation methods
-   Add column alias handling to `validate_option_data`

### Fixes

-   Update example parquet file path
-   Correctly calculate count, DTE, and required capital
-   Remove extra closing brace in cross-validation metric processing

### Refactors

-   Import option data validator from relative path
-   Rename `position_side` to `strategy_side` in option strategies
-   Update butterfly strategy and required capital calculation
-   Improve numeric metric aggregation with robust NaN handling

### Tests

-   Add assertions to validate option data after validation
-   Add required capital tests for debit straddle, butterfly, condor
-   Add `required_capital` tests for debit strategies
-   Add print statement for entry net premium in naked call test
-   Split `test_required_capital` into multiple tests

### Documentation

-   Add `CHANGELOG.md`
