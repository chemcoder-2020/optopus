from configparser import ConfigParser
import pandas as pd
from typing import Dict, Any
import importlib
import inspect
import ast  # Import ast for literal_eval
from loguru import logger


# NOTE: Wildcard imports below are kept for dynamic class loading in
# _parse_condition_section. Flake8 warnings F403 (wildcard) and F401 (unused)
# are suppressed as the imports *are* used dynamically via importlib.
from ..trades.entry_conditions import *  # noqa: F403, F401
from ..trades.external_entry_conditions import *  # noqa: F403, F401
from ..trades.exit_conditions import *  # noqa: F403, F401
from ..trades.option_manager import Config


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

        # Try numeric conversions
        try:
            # Attempt integer conversion first if it looks like an integer
            if param.isdigit() or (param.startswith("-") and param[1:].isdigit()):
                return int(param)
            # Otherwise, try float conversion
            return float(param)
        except ValueError:
            pass  # Not a simple number

        # Handle date/datetime values
        # if "_date" in key.lower():
        #     try:
        #         return pd.Timestamp(param)
        #     except ValueError:
        #         logger.warning(
        #             f"Could not parse '{param}' as Timestamp for key '{key}'. "
        #             f"Returning raw string."
        #         )
        #         return param

        try:
            # Attempt to parse as a timedelta
            return pd.Timedelta(param)
        except ValueError:
            try:
                # Attempt to parse as a date
                return pd.Timestamp(param)
            except ValueError:
                logger.warning(
                    f"Could not parse '{param}' as Timedelta or Timestamp for key '{key}'. "
                )

        # Handle timedelta values based on key naming convention
        # Checks for '_time_' or ends with '_duration' or '_timedelta'
        # timedelta_keys = ("_duration", "_timedelta")
        # if "_time_" in key.lower() or key.lower().endswith(timedelta_keys):
        #     try:
        #         # Attempt to parse using pandas Timedelta
        #         return pd.Timedelta(param)
        #     except ValueError:
        #         logger.warning(
        #             f"Could not parse '{param}' as Timedelta for key '{key}'. "
        #             f"Returning raw string."
        #         )
        #         return param

        # Attempt ast literal_eval
        try:
            evaluated = ast.literal_eval(param)
            return evaluated
        except (ValueError, SyntaxError):
            logger.warning(
                f"Could not evaluate '{param}' as a literal for key '{key}'. "
                f"Returning raw string."
            )
            return param

        # Handle specific time list format (assuming it's a list of time strings)
        # if "allowed_times" in key.lower() or "allowed_days" in key.lower():
        #     try:
        #         # Safely evaluate the string representation of a list
        #         evaluated = ast.literal_eval(param)
        #         if isinstance(evaluated, list):
        #             return evaluated
        #         else:
        #             logger.warning(
        #                 f"Expected a list for key '{key}', but got "
        #                 f"{type(evaluated)}. Returning raw string."
        #             )
        #             return param
        #     except (ValueError, SyntaxError):
        #         logger.warning(
        #             f"Could not evaluate '{param}' as a list for key '{key}'. "
        #             f"Returning raw string."
        #         )
        #         return param

        # Try numeric conversions
        # try:
        #     # Attempt integer conversion first if it looks like an integer
        #     if param.isdigit() or (param.startswith("-") and param[1:].isdigit()):
        #         return int(param)
        #     # Otherwise, try float conversion
        #     return float(param)
        # except ValueError:
        #     pass  # Not a simple number

        # Handle boolean values
        # if param.lower() in ["true", "false"]:
        #     return param.lower() == "true"

        # Evaluate tuple-like expressions using ast.literal_eval for safety
        # if param.startswith("(") and param.endswith(")"):
        #     try:
        #         return ast.literal_eval(param)
        #     except (ValueError, SyntaxError):
        #         # If it's not a valid literal tuple, return as string
        #         logger.warning(
        #             f"Could not evaluate '{param}' as a tuple for key '{key}'. "
        #             f"Returning raw string."
        #         )
        #         return param

        # Default: return the raw string if no other type matches
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

        # Dynamically import class using importlib
        cls = None
        if condition_class:
            try:
                if section == "EXIT_CONDITION":
                    module_path = "optopus.trades.exit_conditions"
                elif section == "ENTRY_CONDITION":
                    module_path = "optopus.trades.entry_conditions"
                elif section == "EXTERNAL_ENTRY_CONDITION":
                    module_path = "optopus.trades.external_entry_conditions"
                else:
                    raise ValueError(f"Unknown section: {section}")

                module = importlib.import_module(module_path)
                class_members = dict(inspect.getmembers(module, inspect.isclass))
                cls = class_members[condition_class]

            except (ValueError, AttributeError, ModuleNotFoundError, KeyError) as e:
                available = list(class_members.keys())
                logger.error(
                    f"Failed to import '{condition_class}' from {module_path}: "
                    f"{e}. Available: {available}. Trying custom module."
                )
                # Attempt import from custom module path
                custom_module_path = ""
                try:
                    if section == "EXIT_CONDITION":
                        custom_module_path = "exit_condition"
                    elif section == "ENTRY_CONDITION":
                        custom_module_path = "entry_condition"
                    elif section == "EXTERNAL_ENTRY_CONDITION":
                        custom_module_path = "external_entry_condition"
                    else:
                        # This case should ideally not be reached if the first try failed
                        raise ValueError(
                            f"Unknown section for custom import: {section}"
                        )

                    module = importlib.import_module(custom_module_path)
                    class_members = dict(inspect.getmembers(module, inspect.isclass))
                    cls = class_members[condition_class]
                except (
                    ValueError,
                    AttributeError,
                    ModuleNotFoundError,
                    KeyError,
                ) as e_custom:
                    available = list(class_members.keys())
                    raise ValueError(
                        f"Failed to import '{condition_class}' from custom module "
                        f"'{custom_module_path}': {e_custom}. Available: {available}"
                    )

        return {"class": cls, "params": params}
