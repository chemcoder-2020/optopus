from ..trades.strategies.iron_condor import IronCondor
from loguru import logger
from .base_backtest import BaseBacktest


class BacktestIronCondor(BaseBacktest):
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
        Initialize the BacktestIronCondor class.

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
        Create an iron condor for the given time and option chain.

        Parameters:
        - time (datetime): The time at which the spread is created.
        - option_chain_df (pd.DataFrame): The option chain DataFrame.

        Returns:
        - OptionStrategy or None: The created iron condor or None if an error occurs.
        """
        try:
            new_spread = IronCondor.create_iron_condor(
                symbol=self.symbol,
                put_long_strike=self.strategy_params["put_long_strike"],
                put_short_strike=self.strategy_params["put_short_strike"],
                call_short_strike=self.strategy_params["call_short_strike"],
                call_long_strike=self.strategy_params["call_long_strike"],
                expiration=self.strategy_params["dte"],
                contracts=self.strategy_params["contracts"],
                entry_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                option_chain_df=option_chain_df,
                profit_target=self.strategy_params.get("profit_target"),
                stop_loss=self.strategy_params.get("stop_loss"),
                commission=self.strategy_params.get("commission", 0.5),
                max_extra_days=self.strategy_params.get("max_extra_days", None),
                exit_scheme=self.strategy_params.get("exit_scheme"),
            )
            return new_spread
        except Exception as e:
            if self.debug:
                logger.error(f"Error creating new iron condor: {e} at {time}")
            return None
