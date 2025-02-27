from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING, List, Tuple
from loguru import logger


if TYPE_CHECKING:
    from .option_manager import OptionBacktester


class EntryConditionChecker(ABC):
    """
    Abstract base class for entry condition checkers.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Check if the entry conditions are met for the option strategy.
    """

    @abstractmethod
    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        pass


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass


class CompositeEntryCondition(EntryConditionChecker):
    """
    Combines multiple entry conditions.

    Args:
        conditions (List[EntryConditionChecker]): List of entry conditions to combine.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if all combined entry conditions are met.
    """

    def __init__(self, conditions: List[EntryConditionChecker]):
        self.conditions = conditions

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        logger.info(f"Checking {len(self.conditions)} composite conditions")

        for i, condition in enumerate(self.conditions, 1):
            condition_name = condition.__class__.__name__
            if not condition.should_enter(strategy, manager, time):
                logger.warning(
                    f"Composite condition {i}/{len(self.conditions)} ({condition_name}) failed"
                )
                return False
            logger.info(
                f"Composite condition {i}/{len(self.conditions)} ({condition_name}) passed"
            )

        return True


class SequentialPipelineCondition(EntryConditionChecker):
    """Process conditions sequentially with configurable logic operators"""

    LOGIC_MAP = {
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b,
        "XOR": lambda a, b: (a or b) and not (a and b),
        "NAND": lambda a, b: not (a and b),
    }

    def __init__(self, steps: List[Tuple[EntryConditionChecker, str]]):
        """
        Args:
            steps: List of (condition, logic_operator) tuples.
                First condition's operator is ignored
        """
        self.steps = steps

    def should_enter(self, strategy, manager, time) -> bool:
        if not self.steps:
            logger.info("SequentialPipeline: No steps configured, denying entry")
            return False

        # Evaluate first step
        first_condition = self.steps[0][0]
        result = first_condition.should_enter(strategy, manager, time)
        logger.info(
            f"SequentialPipeline Step 1/{len(self.steps)} ({first_condition.__class__.__name__}): {result}"
        )
        if not result:
            if self.steps[0][1] == "AND":
                logger.info("SequentialPipeline: First step failed, denying entry")
                return False
            else:
                logger.info(
                    "SequentialPipeline: First step failed, still evaluating remaining steps"
                )

        # Evaluate remaining steps
        for i, (condition, logic) in enumerate(self.steps[1:], start=2):
            condition_name = condition.__class__.__name__

            if logic not in self.LOGIC_MAP:
                raise ValueError(f"Invalid logic operator: {logic}")

            current = condition.should_enter(strategy, manager, time)
            new_result = self.LOGIC_MAP[logic](result, current)

            logger.info(
                f"SequentialPipeline Step {i}/{len(self.steps)} ({condition_name}) "
                f"with logic '{logic}': {result} {logic} {current} => {new_result}"
            )

            result = new_result

            # Short-circuit evaluation
            if logic == "AND" and not result:
                logger.info("Short-circuiting due to AND logic with False result")
                break
            if logic == "OR" and result:
                logger.info("Short-circuiting due to OR logic with True result")
                break

        logger.info(f"Final SequentialPipeline result: {result}")
        return result


class ConditionalGate(EntryConditionChecker):
    """Only evaluate main condition if pre-condition passes"""

    def __init__(
        self,
        main_condition: EntryConditionChecker,
        pre_condition: EntryConditionChecker,
    ):
        self.main_condition = main_condition
        self.pre_condition = pre_condition

    def should_enter(self, strategy, manager, time) -> bool:
        if self.pre_condition.should_enter(strategy, manager, time):
            return self.main_condition.should_enter(strategy, manager, time)
        return True  # Bypass if pre-condition fails
