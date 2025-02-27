from .base import Preprocessor
from loguru import logger

class PremiumListInit(Preprocessor):
    def preprocess(self, strategy, manager):
        if not hasattr(manager, "context"):
            manager.context = {}

        if "premiums" not in manager.context:
            manager.context["premiums"] = []

        logger.debug("Premium log initialized for manager")
        return True
