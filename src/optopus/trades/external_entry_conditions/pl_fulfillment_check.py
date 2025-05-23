from .base import BaseComponent
import pandas as pd
from loguru import logger


class PLFulfilmentCheck(BaseComponent):
    """
    A component to check profit and loss fulfillment conditions for strategy entry.
    
    This class evaluates whether a strategy should enter based on its performance
    relative to specified return targets and loss prevention thresholds.
    
    Attributes:
        target_return (float): Target return threshold for entry decision
        target_loss_prevention (float): Optional loss prevention threshold
        freq (str): Time frequency for performance evaluation (e.g., 'W' for weekly)
    """
    
    def __init__(
        self,
        target_return=0.01,
        target_loss_prevention=None,
        freq="W",
    ):
        """
        Initialize the PL Fulfilment Check component.
        
        Args:
            target_return (float): Target return threshold for entry decision
            target_loss_prevention (float, optional): Loss prevention threshold to avoid entering
            freq (str): Time frequency for performance evaluation (e.g., 'W' for weekly)
        """
        self.target_return = target_return
        self.target_loss_prevention = target_loss_prevention
        self.freq = freq
        logger.debug(f"Initialized PLFulfilmentCheck with target_return={target_return}, target_loss_prevention={target_loss_prevention}, freq={freq}")

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        """
        Determine if the strategy should enter based on P&L performance.
        
        Args:
            strategy: The trading strategy to evaluate
            manager: The strategy manager
            time: The current timestamp
            
        Returns:
            bool: True if the strategy should enter, False otherwise
        """
        logger.debug("Calculating P&L fulfillment for entry decision")
        
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
            
            closed_pl_weekly = (
                pd.DataFrame(recent_performance)
                .set_index("time")["closed_pl"]
                .resample(self.freq)
                .last()
            )
            if len(closed_pl_weekly) < 2:
                logger.info("Trading history is too short for P&L fulfillment check. Entering based on other conditions.")
                return True
            else:
                pl = closed_pl_weekly.diff()
            current_pl = pl.iloc[-1]
    
            current_return = current_pl / manager.config.initial_capital
            logger.info(f"Current {self.freq} return: {current_return:.4f}")
            
            manager.context["indicators"].update(
                {f"Current {self.freq} Return": current_return}
            )
    
            if self.target_loss_prevention is not None:
                should_enter_flag = -self.target_loss_prevention < current_return < self.target_return
                logger.debug(f"Entry decision with loss prevention: {should_enter_flag} (Return: {current_return:.4f}, Loss Prevention Threshold: {-self.target_loss_prevention:.4f}, Target Return: {self.target_return:.4f})")
            else:
                should_enter_flag = current_return < self.target_return
                logger.debug(f"Entry decision without loss prevention: {should_enter_flag} (Return: {current_return:.4f}, Target Return: {self.target_return:.4f})")
                
            return should_enter_flag
            
        except Exception as e:
            logger.error(f"Error in P&L fulfillment check: {str(e)}")
            return False
