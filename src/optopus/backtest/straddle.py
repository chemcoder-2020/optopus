from ..trades.strategies.straddle import Straddle
from loguru import logger
from .base_backtest import BaseBacktest


class BacktestStraddle(BaseBacktest):
    def __init__(
        self,
        config,
        data_folder,
        start_date,
        end_date,
        trading_start_time,
        trading_end_time,
        strategy_params,
        debug=False,
    ):
        """
        Initialize the BacktestStraddle class.

        Parameters:
        - config (dict): Configuration parameters.
        - data_folder (str): Path to the data folder.
        - start_date (str): Start date for the backtest.
        - end_date (str): End date for the backtest.
        - trading_start_time (str): Start time for trading.
        - trading_end_time (str): End time for trading.
        - strategy_params (dict): Strategy parameters.
        - debug (bool, optional): Debug mode flag. Defaults to False.
        """
        super().__init__(
            config,
            data_folder,
            start_date,
            end_date,
            trading_start_time,
            trading_end_time,
            debug=debug,
        )
        self.strategy_params = strategy_params
        self.symbol = self.strategy_params["symbol"]

    def create_spread(self, time, option_chain_df):
        """
        Create a straddle for the given time and option chain.

        Parameters:
        - time (datetime): The time at which the spread is created.
        - option_chain_df (pd.DataFrame): The option chain DataFrame.

        Returns:
        - OptionStrategy or None: The created straddle or None if an error occurs.
        """
        try:
            new_spread = Straddle.create_straddle(
                symbol=self.symbol,
                strike=self.strategy_params["strike"],
                expiration=self.strategy_params["dte"],
                contracts=self.strategy_params["contracts"],
                entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                option_chain_df=option_chain_df,
                profit_target=self.strategy_params.get("profit_target"),
                stop_loss=self.strategy_params.get("stop_loss"),
                commission=self.strategy_params.get("commission", 0.5),
                max_extra_dte=self.strategy_params.get("max_extra_dte", None),
                exit_scheme=self.strategy_params.get("exit_scheme"),
                strategy_side=self.strategy_params.get("strategy_side", "DEBIT"),
            )
            return new_spread
        except Exception as e:
            if self.debug:
                logger.error(f"Error creating new spread: {e} at {time}")
            return None
