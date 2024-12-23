# Optopus

Optopus is an option trading library designed to support multiple brokers and perform backtesting. Currently, it supports a file-based backtesting system. The library provides various functionalities for option trading, including:

- **Data Processing**: Includes methods to format and process raw option chain data into structured formats like Pandas DataFrames, making it easier to work with the data.
- **Option Strategies**: Supports the creation and management of option strategies, including naked calls, vertical spreads, iron condors, and other complex strategies.
- **Backtesting**: Provides a framework for backtesting option strategies using historical data.
- **Entry and Exit Conditions**: Defines customizable entry and exit conditions for trades, including capital requirements, position limits, return over risk thresholds, and conflict checks.
- **Option Chain Converter**: Converts raw option chain data into a structured format, allowing for easy selection of strikes based on various criteria such as delta, strike price, and ATM offsets.

### Key Components

- **OptionLeg Class**:
  - Represents a single option leg in an options trading strategy.
  - Provides methods to update the leg's status and calculate profit and loss.

- **OptionStrategy Class**:
  - Represents an options trading strategy composed of multiple option legs.
  - Provides methods to add legs, update the strategy, and calculate total profit and loss.

- **OptionBacktester Class**:
  - Manages the backtesting of option strategies.
  - Initializes with a configuration and manages capital, active trades, and closed trades.

- **Entry Conditions**:
  - **CapitalRequirementCondition**: Checks if there is sufficient capital for the trade.
  - **PositionLimitCondition**: Checks if position limits are respected.
  - **RORThresholdCondition**: Checks if return over risk meets the threshold.
  - **ConflictCondition**: Checks for conflicts with existing positions.
  - **CompositeEntryCondition**: Combines multiple entry conditions.
  - **DefaultEntryCondition**: Combines all standard checks.

- **Exit Conditions**:
  - **ProfitTargetCondition**: Exit condition based on a profit target.
  - **StopLossCondition**: Exit condition based on a stop loss.
  - **TimeBasedCondition**: Exit condition based on a specific time before expiration.
  - **TrailingStopCondition**: Exit condition based on a trailing stop.
  - **CompositeExitCondition**: Composite exit condition that combines multiple exit conditions.
  - **DefaultExitCondition**: Default exit condition that combines a profit target and a time-based condition.

- **Option Chain Converter**:
  - Converts raw option chain data into a structured format.
  - Provides methods to get the closest expiration date, ATM strike, and desired strike based on various criteria.

### Example Usage

1. **Creating an Option Strategy**:
   ```python
   from optopus.trades.option_spread import OptionStrategy, OptionLeg

   strategy = OptionStrategy(symbol="SPY", strategy_type="Iron Condor")
   strategy.add_leg(OptionLeg(symbol="SPY", option_type="CALL", strike=450, expiration=datetime(2024, 10, 1), contracts=1, action="SELL"))
   strategy.add_leg(OptionLeg(symbol="SPY", option_type="CALL", strike=455, expiration=datetime(2024, 10, 1), contracts=1, action="BUY"))
   strategy.add_leg(OptionLeg(symbol="SPY", option_type="PUT", strike=440, expiration=datetime(2024, 10, 1), contracts=1, action="SELL"))
   strategy.add_leg(OptionLeg(symbol="SPY", option_type="PUT", strike=435, expiration=datetime(2024, 10, 1), contracts=1, action="BUY"))
   print(strategy.total_pl())
   ```

2. **Running a Backtest**:
   ```python
    from optopus.backtest.naked_call import BacktestNakedCall
    from loguru import logger
    
    from config import (
        DATA_FOLDER,
        START_DATE,
        END_DATE,
        TRADING_START_TIME,
        TRADING_END_TIME,
        DEBUG,
        STRATEGY_PARAMS,
        BACKTESTER_CONFIG,
    )
    
    logger.disable("optopus")
    
    backtest = BacktestNakedCall(
        config=BACKTESTER_CONFIG,
        data_folder=DATA_FOLDER,
        start_date=START_DATE,
        end_date=END_DATE,
        trading_start_time=TRADING_START_TIME,
        trading_end_time=TRADING_END_TIME,
        strategy_params=STRATEGY_PARAMS,
        debug=DEBUG,
    )
    bt = backtest.run_backtest()
    closed_trades_df = bt.get_closed_trades_df().set_index("exit_time").sort_index()
    print(closed_trades_df)
   ```

3. **Using the Option Chain Converter**:
    ```python
    import pandas as pd
    from datetime import datetime
    from optopus.trades.option_chain_converter import OptionChainConverter

    # Load the test data
    file_path = "path/to/your/option_chain_data.parquet"
    option_chain_df = pd.read_parquet(file_path)

    # Initialize the OptionChainConverter
    converter = OptionChainConverter(option_chain_df)

    # Example usage of get_closest_expiration
    target_date_int = 30  # 30 days from QUOTE_READTIME
    closest_expiration_int = converter.get_closest_expiration(target_date_int)
    print(f"Closest expiration (int): {closest_expiration_int}")

    # Example usage of get_desired_strike with different expiration types
    target_delta = 0.30
    expiry_as_dte = 30  # 30 days from now

    # Using integer (DTE)
    call_strike_dte = converter.get_desired_strike(
        expiry_as_dte, "CALL", target_delta, by="delta"
    )
    put_strike_dte = converter.get_desired_strike(
        expiry_as_dte, "PUT", target_delta, by="delta"
    )
    print(f"Using DTE ({expiry_as_dte}):")
    print(f"  CALL strike at delta {target_delta}: {call_strike_dte}")
    print(f"  PUT strike at delta {target_delta}: {put_strike_dte}")
    ```

### Future Plans

- **Support for Additional Brokers**: The library plans to support more brokers in the future, expanding its utility for a broader range of users.
- **Enhanced Data Processing**: Additional methods for processing and analyzing market data.
- **Advanced Trading Strategies**: Implementation of more complex trading strategies and risk management tools.

This project is designed to be modular and extensible, making it easy to add new features and support for additional brokers. Contributions are welcome!

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chemcoder-2020/optopus.git
   cd optopus
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

5. Alternatively, you can install the library using the binary distribution:
   ```bash
   pip install dist/optopus-0.9.1-dev5-py3-none-any.whl
   ```

## Usage

Run the setup script to create a backtesting project:
```bash
setup-optopus-backtest project_name --symbol SPY --dte 45 --strike ATM --contracts 1000 --commission 0.5 --initial_capital 20000 --max_positions 10 --position_size 0.1
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
