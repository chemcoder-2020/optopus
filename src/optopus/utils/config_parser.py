from configparser import ConfigParser
import pandas as pd
from typing import Dict, Any
import importlib
import difflib
import inspect
from ..trades.option_manager import Config
from ..trades.entry_conditions import *
from ..trades.external_entry_conditions import *
from ..trades.exit_conditions import *
from loguru import logger


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
        """Parse entry/external entry condition/exit condition configuration"""
        if not self.parser.has_section(section):
            return {"class": None, "params": {}}

        params = {
            k: self._parse_value(k, v) for k, v in self.parser.items(section, raw=True)
        }
        condition_class = params.pop("class", None)
        
        if not condition_class:
            return {"class": None, "params": params}

        # Map sections to their base classes and modules
        SECTION_CONFIG = {
            "EXIT_CONDITION": {
                "module": "optopus.trades.exit_conditions.base",
                "base_class": "ExitConditionChecker"
            },
            "ENTRY_CONDITION": {
                "module": "optopus.trades.entry_conditions.base", 
                "base_class": "EntryConditionChecker"
            },
            "EXTERNAL_ENTRY_CONDITION": {
                "module": "optopus.trades.external_entry_conditions.base",
                "base_class": "ExternalEntryConditionChecker"
            }
        }

        valid_classes = []
        suggestions = []
        module_path = None
        base_class = None

        try:
            if section not in SECTION_CONFIG:
                raise ValueError(f"Unsupported configuration section: {section}")

            config = SECTION_CONFIG[section]
            module_path = config["module"]
            base_class_name = config["base_class"]

            # Import base class module
            base_module = importlib.import_module(module_path)
            base_class = getattr(base_module, base_class_name)

            # Try importing from standard module first
            module = importlib.import_module(module_path.replace(".base", ""))
            cls = getattr(module, condition_class)
            
            # Validate class hierarchy
            if not issubclass(cls, base_class):
                raise TypeError(f"{condition_class} is not a subclass of {base_class_name}")

            valid_classes = [
                name for name, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, base_class) and obj != base_class
            ]

        except Exception as e:
            # If standard import fails, try custom modules
            try:
                custom_module_path = section.lower().replace("_", "")
                module = importlib.import_module(custom_module_path)
                cls = getattr(module, condition_class)
                valid_classes = inspect.getmembers(module, inspect.isclass)
            except Exception as secondary_error:
                # Generate suggestions from valid classes
                if valid_classes:
                    suggestions = difflib.get_close_matches(
                        condition_class,
                        valid_classes,
                        n=3,
                        cutoff=0.6
                    )
                
                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = f" Did you mean: {', '.join(suggestions)}?"
                
                available_msg = ""
                if valid_classes:
                    available_msg = f"\nAvailable options: {', '.join(sorted(valid_classes))}"
                elif module_path:
                    available_msg = f"\nNo valid condition classes found in {module_path}"

                raise ValueError(
                    f"Failed to load condition class '{condition_class}'.{suggestion_msg}"
                    f"{available_msg}\nOriginal error: {str(e)}"
                ) from secondary_error

        return {"class": cls, "params": params}
