from configparser import ConfigParser
import pandas as pd
from typing import Dict, Any
import importlib
from ..trades.option_manager import Config
from ..trades.entry_conditions import DefaultEntryCondition, EntryConditionChecker
from ..trades.external_entry_conditions import ExternalEntryConditionChecker, CompositePipelineCondition
from ..trades.exit_conditions import ExitCondition


class IniConfigParser:
    """Parses INI config files into Config dataclass and strategy parameters"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.parser = ConfigParser()
        self.parser.read(config_path)

    def _parse_value(self, value: str) -> Any:
        """Convert INI string values to appropriate Python types"""
        try:
            # Detect and parse special formats
            if value.startswith("Timedelta("):
                return pd.Timedelta(value.split("(", 1)[1].rstrip(")"))
            if value.startswith("(") and value.endswith(")"):
                return tuple(self._parse_value(v) for v in value[1:-1].split(","))
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            if "." in value:
                return float(value)
            return int(value)
        except (ValueError, AttributeError):
            return value

    def get_strategy_params(self) -> Dict[str, Any]:
        """Parse strategy-specific parameters from [STRATEGY_PARAMS] section"""

        params = {}
        if self.parser.has_section("STRATEGY_PARAMS"):
            for key, value in self.parser.items("STRATEGY_PARAMS"):
                params[key] = self._parse_value(value)

        # Parse exit scheme configuration from EXIT_CONDITION section
        if self.parser.has_section("EXIT_CONDITION"):
            exit_config = self._parse_condition_section("EXIT_CONDITION")
            params["exit_scheme"] = {
                "class": exit_config["class"],
                "params": exit_config["params"],
            }

        return params

    def get_config(self) -> Config:
        """Parse and return Config dataclass from INI file"""
        config_params = {}

        # Parse general configuration
        if self.parser.has_section("BACKTESTER_CONFIG"):
            config_params.update(
                {
                    k: self._parse_value(v)
                    for k, v in self.parser.items("BACKTESTER_CONFIG")
                }
            )

        # Parse entry conditions
        config_params["entry_condition"] = self._parse_condition_section(
            "ENTRY_CONDITION"
        )

        # Parse external entry conditions
        config_params["external_entry_condition"] = self._parse_condition_section(
            "EXTERNAL_ENTRY_CONDITION"
        )

        return Config(**config_params)

    def _parse_condition_section(self, section: str) -> Dict[str, Any]:
        """Parse entry/external entry condition configuration"""
        if not self.parser.has_section(section):
            return {"class": None, "params": {}}

        params = {k: self._parse_value(v) for k, v in self.parser.items(section)}
        condition_class = params.pop("class", None)

        # Safely evaluate class reference

        try:
            cls = eval(condition_class) if condition_class else None
        except (NameError, SyntaxError):
            raise ValueError(f"Invalid class reference: {condition_class}")

        return {"class": cls, "params": params}

    @staticmethod
    def test():
        """Example usage/test"""
        parser = IniConfigParser("config.ini")

        config = parser.get_config()
        print("Config:", config)

        strategy_params = parser.get_strategy_params()
        print("Strategy Params:", strategy_params)
