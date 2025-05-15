from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union
import datetime


class PLCheckForExit(BaseComponent):
    """
    A component to check profit and loss conditions for strategy exit.
    
    This class evaluates whether a strategy should exit based on its performance
    relative to specified return targets and loss prevention thresholds at a given frequency.
    
    Attributes:
        target_return (float): Target return threshold for exit decision
        target_loss_prevention (float, optional): Loss prevention threshold for exit decision. Defaults to None.
        loss_prevention_to_last_period (float, optional): Multiplier for comparing current performance to previous period. Defaults to None.
        freq (str): Time frequency for performance evaluation (e.g., 'W' for weekly, 'D' for daily)
    """
    
    def __init__(
        self,
        target_return=0.01,
        target_loss_prevention=None,
        loss_prevention_to_last_period=None,
        freq="W",
    ):
        """
        Initialize the PL Check for Exit component.
        
        Args:
            target_return (float): Target return threshold for exit decision
            target_loss_prevention (float, optional): Loss prevention threshold for exit decision. If None, this check is skipped.
            loss_prevention_to_last_period (float, optional): Multiplier for comparing current performance to previous period. If None, this check is skipped.
            freq (str): Time frequency for performance evaluation (e.g., 'W' for weekly, 'D' for daily)
        """
        self.target_return = target_return
        self.target_loss_prevention = target_loss_prevention
        self.loss_prevention_to_last_period = loss_prevention_to_last_period
        self.freq = freq
        logger.debug(f"Initialized PLCheckForExit with target_return={target_return}, target_loss_prevention={target_loss_prevention}, loss_prevention_to_last_period={loss_prevention_to_last_period}, freq='{freq}'")

    def should_exit(self, strategy, manager, time: pd.Timestamp) -> bool:
        """
        Determine if the strategy should exit based on P&L performance.
        
        Args:
            strategy: The trading strategy to evaluate
            manager: The strategy manager
            time: The current timestamp
            
        Returns:
            bool: True if the strategy should exit, False otherwise
        """
        logger.debug(f"Calculating P&L fulfillment for exit decision at frequency '{self.freq}'")
        
        try:
            # Estimate window size based on frequency
            if self.freq == "W":  # Weekly
                window = 26 * 7 * 3  # 1 week at 15-min intervals (26 15-min intervals per hour * 7 hours)
            elif self.freq == "D":  # Daily
                window = 26 * 1 * 3  # 1 day at 15-min intervals
            elif self.freq == "M":  # Monthly
                window = 26 * 30 * 3  # 1 month at 15-min intervals
            else:  # Default for unknown frequencies
                window = 26 * 7 * 3  # 1 week at 15-min intervals
            
            # Slice performance_data to only relevant portion before DataFrame conversion
            recent_performance = manager.performance_data[-window:]
            
            pl = (
                pd.DataFrame(recent_performance)
                .set_index("time")["total_pl"]
                .resample(self.freq)
                .last()
                .diff()
            )
            
            if len(pl) < 2:
                logger.warning(f"Not enough performance data for {self.freq} resampling. Need at least 2 periods.")
                return False
                
            previous_pl = pl.iloc[-2]
            current_pl = pl.iloc[-1]
            current_return = current_pl / manager.config.initial_capital

            # Prevention check: exit if current PL loss exceeds previous period's gain
            loss_prevention_triggered = False
            if self.loss_prevention_to_last_period is not None and previous_pl > 0:
                loss_prevention_triggered = current_pl < -previous_pl * self.loss_prevention_to_last_period
                logger.debug(f"Loss prevention check: current_pl={current_pl:.2f}, previous_pl={previous_pl:.2f}, threshold={-previous_pl * self.loss_prevention_to_last_period:.2f}, triggered={loss_prevention_triggered}")

            # Update context with current metrics
            manager.context["indicators"].update({
                f"Current {self.freq} PL": current_pl,
                f"Current {self.freq} Return": current_return
            })

            # Determine exit condition
            exit_conditions = []
            if self.target_loss_prevention is not None:
                exit_conditions.append(-self.target_loss_prevention > current_return)
            
            exit_conditions.append(current_return > self.target_return)
            
            if self.loss_prevention_to_last_period is not None:
                exit_conditions.append(loss_prevention_triggered)
            
            exit_condition = any(exit_conditions)
            
            logger.debug(f"Exit condition evaluation: return={current_return:.4f}, target_return={self.target_return:.4f}, target_loss_prevention={self.target_loss_prevention}, loss_prevention_triggered={loss_prevention_triggered if self.loss_prevention_to_last_period is not None else 'skipped'}, triggered={exit_condition}")
            
            return exit_condition
            
        except Exception as e:
            logger.error(f"Error in P&L fulfillment check: {str(e)}")
            return False
