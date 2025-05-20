# MODULE TO DEFINE CUSTOM EXIT CONDITIONS
from optopus.trades.exit_conditions import (
    ProfitTargetCondition,
    TimeBasedCondition,
    StopLossCondition,
    ExitConditionChecker,
    CompositePipelineCondition,
    PremiumListInit,
    PremiumFilter,
)
from typing import Union
import pandas as pd
from datetime import datetime

# Uncomment the below class if you want to use the ProfitStoplossExitCondition

# class ProfitStoplossExitCondition(ExitConditionChecker):
#     def __init__(
#         self, profit_target: float, exit_time_before_expiration: pd.Timedelta, **kwargs
#     ):
#         self.flow = CompositePipelineCondition(
#             pipeline=ProfitTargetCondition(profit_target)
#             | TimeBasedCondition(exit_time_before_expiration)
#             | StopLossCondition(stop_loss=kwargs.get("stop_loss", 100)),
#             preprocessors=[
#                 PremiumListInit(),
#                 PremiumFilter(
#                     filter_method=kwargs.get("filter_method"),
#                     window_size=kwargs.get("window_size"),
#                     n_sigma=kwargs.get("n_sigma", 3),
#                     k=kwargs.get("k", 1.4826),
#                     max_iterations=kwargs.get("max_iterations", 5),
#                     replace_with_na=kwargs.get("replace_with_na", True),
#                     implementation=kwargs.get("implementation", "pandas"),
#                 ),
#             ],
#             **kwargs,
#         )

#     def should_exit(
#         self,
#         strategy,
#         current_time: Union[datetime, str, pd.Timestamp],
#         option_chain_df: pd.DataFrame,
#     ) -> bool:
#         return self.flow.should_exit(
#             strategy=strategy,
#             current_time=current_time,
#             option_chain_df=option_chain_df,
#         )
