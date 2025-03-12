from .base import Preprocessor
from loguru import logger


class RORListInit(Preprocessor):
    def preprocess(self, strategy, manager):
        if not hasattr(manager, "context"):
            manager.context = {}

        if "RoRs" not in manager.context:
            manager.context["RoRs"] = []

        logger.debug("RoRs log initialized for manager")
        return True
