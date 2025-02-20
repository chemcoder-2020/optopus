from configparser import ConfigParser
import pandas as pd
from typing import Dict, Any
import importlib
from ..trades.option_manager import Config
from ..trades.entry_conditions import *
from ..trades.external_entry_conditions import *
from ..trades.exit_conditions import *


class IniConfigParser:
    """Parses INI config files into Config dataclass and strategy parameters"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.parser = ConfigParser()
        self.parser.read(config_path)

    def _parse_value(self, key: str, value: str) -> Any:
        """Convert INI string values to appropriate Python types"""
        param = value

        # Preserve strike/delta values as raw strings
        if "delta" in key or "strike" in key:
            return param

        # Handle date/datetime values
        if "_date" in key.lower():
            return pd.Timestamp(param)

        # Handle specific time formats
        if "exit_time_before_expiration" in key.lower():
            return pd.Timedelta(param)

        if "allowed_times" in key.lower():
            return eval(param)

        # Try numeric conversions
        if "." in param:
            try:
                return float(param)
            except ValueError:
                pass
        elif param.isnumeric():
            return int(param)

        # Handle boolean values
        if param.lower() in ["true", "false"]:
            return param.lower() == "true"

        # Evaluate tuple-like expressions using ast.literal_eval
        if "(" in param and ")" in param:
            try:
                return eval(param)
            except:
                return param

        return param

    def get_strategy_params(self) -> Dict[str, Any]:
        """Parse strategy-specific parameters from [STRATEGY_PARAMS] section"""

        params = {}
        if self.parser.has_section("STRATEGY_PARAMS"):
            for key, value in self.parser.items("STRATEGY_PARAMS", raw=True):
                params[key] = self._parse_value(key, value)

        # Parse exit scheme configuration from EXIT_CONDITION section
        if self.parser.has_section("EXIT_CONDITION"):
            exit_config = self._parse_condition_section("EXIT_CONDITION")
            params["exit_scheme"] = {
                "class": exit_config["class"],
                "params": exit_config["params"],
            }

        return params

    def get_general_params(self) -> Dict[str, Any]:
        """Parse general configuration parameters from [GENERAL] section"""
        params = {}
        if self.parser.has_section("GENERAL"):
            for key, value in self.parser.items("GENERAL", raw=True):
                params[key] = self._parse_value(key, value)
        return params

    def get_config(self) -> Config:
        """Parse and return Config dataclass from INI file"""
        config_params = {}

        # Parse general configuration
        if self.parser.has_section("BACKTESTER_CONFIG"):
            config_params.update(
                {
                    k: self._parse_value(k, v)
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

        params = {
            k: self._parse_value(k, v) for k, v in self.parser.items(section, raw=True)
        }
        condition_class = params.pop("class", None)

        # Safely evaluate class reference

        try:
            cls = eval(condition_class) if condition_class else None
        except (NameError, SyntaxError):
            raise ValueError(f"Invalid class reference: {condition_class}")

        return {"class": cls, "params": params}
