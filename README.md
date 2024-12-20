# Optopus

Optopus is an option trading library designed to support multiple brokers. Currently, it supports Schwab. The library provides various functionalities for option trading, including:

- **Broker Integration**: Integrates with the Schwab API to fetch and process market data, such as option chains, equity quotes, and price history.
- **Data Processing**: Includes methods to format and process the raw data from the API into structured formats like Pandas DataFrames, making it easier to work with the data.
- **Option Strategies**: Supports the creation and management of option strategies, including vertical spreads and other complex strategies.
- **Market Hours and Status**: Provides methods to check market hours and whether the market is open for a given date and time.
- **Price History**: Fetches historical price data for equities, which can be used for backtesting and analysis.

### Key Components

- **SchwabData Class**:
  - **Initialization**: Sets up the Schwab API client with necessary credentials and base URLs.
  - **Option Chain**: Fetches and processes option chains for a given symbol.
  - **Market Hours**: Retrieves market hours for a specified market.
  - **Quote Retrieval**: Fetches and formats equity and option quotes.
  - **Price History**: Fetches historical price data for a given symbol.

- **OptionLeg Class**:
  - Represents a single option leg in an options trading strategy.
  - Provides methods to update the leg's status and calculate profit and loss.

- **OptionStrategy Class**:
  - Represents an options trading strategy composed of multiple option legs.
  - Provides methods to add legs, update the strategy, and calculate total profit and loss.

- **OptionBacktester Class**:
  - Manages the backtesting of option strategies.
  - Initializes with a configuration and manages capital, active trades, and closed trades.

### Example Usage

1. **Fetching Option Chain**:
   ```python
   from src.optopus.brokers.schwab.schwab_data import SchwabData

   schwab = SchwabData(client_id="your_client_id", client_secret="your_client_secret")
   option_chain = schwab.get_option_chain(symbol="AAPL")
   print(option_chain)
   ```

2. **Fetching Equity Quote**:
   ```python
   equity_quote = schwab.get_quote(symbols="AAPL")
   print(equity_quote)
   ```

3. **Fetching Price History**:
   ```python
   price_history = schwab.get_price_history(symbol="AAPL", period_type="year", period=1, frequency_type="daily", frequency=1)
   print(price_history)
   ```

4. **Creating an Option Strategy**:
   ```python
   from src.optopus.trades.option_spread import OptionStrategy, OptionLeg

   strategy = OptionStrategy(symbol="AAPL", strategy_type="Vertical Spread")
   strategy.add_leg(OptionLeg(symbol="AAPL", option_type="CALL", strike=150, expiration=datetime(2023, 10, 1), contracts=1))
   strategy.add_leg(OptionLeg(symbol="AAPL", option_type="CALL", strike=160, expiration=datetime(2023, 10, 1), contracts=1))
   print(strategy.total_pl())
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
   pip install dist/optopus-0.4.2-py3-none-any.whl
   ```

## Usage

Run the project:
```bash
my_python_project
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
