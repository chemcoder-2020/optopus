from .base import BaseComponent
import pandas as pd
from loguru import logger
from ...utils.option_chain_features import compare_near_atm_prices


class OptionPriceCheck(BaseComponent):
    def __init__(self, target_dte=0, comparison_range=0) -> None:
        class_name = self.__class__.__name__
        try:
            self.target_dte = target_dte
            self.comparison_range = comparison_range
            self.name = f"C-P Difference_{target_dte}_{comparison_range}"
        except Exception as e:
            logger.error(f"{class_name}.__init__ failed: {e}", exc_info=True)
            raise

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        class_name = self.__class__.__name__

        try:
            if not hasattr(manager, "context"):
                logger.warning(
                    f"{class_name} [{time}]: manager has no 'context' attribute."
                )
                return False

            option_chain_df = manager.context.get("option_chain_df")
            if option_chain_df is None:
                logger.warning(
                    f"{class_name} [{time}]: 'option_chain_df' is None in "
                    f"manager.context."
                )
                return False
            if not isinstance(option_chain_df, pd.DataFrame) or option_chain_df.empty:
                logger.warning(
                    f"{class_name} [{time}]: 'option_chain_df' is not a DataFrame or "
                    f"is empty."
                )
                return False

            option_chain_df_copy = (
                option_chain_df.copy()
            )  #  Avoid SettingWithCopyWarning
            option_chain_df_copy["C_MARK"] = (
                option_chain_df_copy["C_BID"] + option_chain_df_copy["C_ASK"]
            ) / 2
            option_chain_df_copy["P_MARK"] = (
                option_chain_df_copy["P_BID"] + option_chain_df_copy["P_ASK"]
            ) / 2

            price_comparison = compare_near_atm_prices(
                option_chain_df_copy,
                target_dte=self.target_dte,
                comparison_range=self.comparison_range,
            )

            if price_comparison is None:
                logger.info(
                    f"{class_name} [{time}]: compare_near_atm_prices returned None."
                )
                return False
            if not isinstance(price_comparison, pd.DataFrame) or price_comparison.empty:
                logger.info(
                    f"{class_name} [{time}]: price_comparison DataFrame is empty or "
                    f"not a DataFrame."
                )
                return False
            if "Difference (C-P)" not in price_comparison.columns:
                logger.error(
                    f"{class_name} [{time}]: 'Difference (C-P)' not in "
                    f"price_comparison DataFrame."
                )
                return False

            average_price_difference = price_comparison["Difference (C-P)"].mean()
            if "indicators" not in manager.context:
                manager.context["indicators"] = {}
            manager.context["indicators"].update({self.name: average_price_difference})

            if not hasattr(manager, "performance_data") or not manager.performance_data:
                logger.warning(
                    f"{class_name} [{time}]: manager.performance_data is missing or "
                    f"empty."
                )
                return False

            data_slice = manager.performance_data[-26:]
            if not data_slice:
                log_msg = (
                    f"{class_name} [{time}]: Not enough performance_data for "
                    f"26-period lookback, available: "
                    f"{len(manager.performance_data)}."
                )
                logger.warning(log_msg)
                return False

            data = pd.DataFrame(data_slice).reset_index(drop=True)
            if "indicators" not in data.columns:
                logger.error(
                    f"{class_name} [{time}]: 'indicators' column not found in "
                    f"performance_data DataFrame slice."
                )
                return False

            valid_indicators_data = data[
                data["indicators"].apply(lambda x: isinstance(x, dict))
            ].copy()
            if valid_indicators_data.empty:
                logger.warning(
                    f"{class_name} [{time}]: No valid dict indicators found in "
                    f"performance_data slice."
                )
                return False

            # Convert Series of dicts to list of dicts for json_normalize
            indicators_list = valid_indicators_data["indicators"].tolist()
            cp_diff_normalized = pd.json_normalize(indicators_list)

            if self.name not in cp_diff_normalized.columns:
                log_msg = (
                    f"{class_name} [{time}]: Historical indicator '{self.name}' "
                    f"not available in normalized performance data."
                )
                logger.warning(log_msg)
                return False

            cp_diff = pd.concat(
                [
                    cp_diff_normalized,
                    valid_indicators_data.drop(columns=["indicators"]),
                ],
                axis=1,
            ).set_index("time")[self.name]

            if cp_diff.empty:
                logger.warning(
                    f"{class_name} [{time}]: cp_diff series for '{self.name}' is "
                    f"empty after processing."
                )
                return False

            if not isinstance(cp_diff.index, pd.DatetimeIndex) or cp_diff.index.empty:
                logger.error(
                    f"{class_name} [{time}]: cp_diff.index for '{self.name}' is not "
                    f"a DatetimeIndex or is empty."
                )
                return False

            last_timestamp_date = pd.Timestamp(cp_diff.index[-1].date())
            todays_cp_diff = cp_diff[cp_diff.index >= last_timestamp_date]

            if todays_cp_diff.empty:
                log_msg = (
                    f"{class_name} [{time}]: No 'todays_cp_diff' for '{self.name}' "
                    f"found for {last_timestamp_date}."
                )
                logger.info(log_msg)
                return False

            open_cp_diff = todays_cp_diff.iloc[0]

            logger.info(
                f"{class_name} ({self.name}) [{time}]: Open CP Diff: "
                f"{open_cp_diff:.3f}"
            )
            logger.info(
                f"{class_name} ({self.name}) [{time}]: Current CP Diff: "
                f"{average_price_difference:.3f}"
            )

            dropping_cp_diff = open_cp_diff > average_price_difference > 0

            if dropping_cp_diff:
                logger.info(f"{class_name} Passed ({self.name}) [{time}].")
                return True
            else:
                log_msg = (
                    f"{class_name} Failed ({self.name}) [{time}]: "
                    f"Open={open_cp_diff:.3f}, "
                    f"Current={average_price_difference:.3f}."
                )
                logger.info(log_msg)
                return False

        except KeyError as e:
            context_keys_str = "No context"
            if hasattr(manager, "context") and manager.context is not None:
                context_keys_str = str(list(manager.context.keys()))
            log_msg = (
                f"{class_name}.should_enter [{time}]: KeyError - {str(e)}. "
                f"Context keys: {context_keys_str}."
            )
            logger.error(log_msg, exc_info=True)
            return False
        except IndexError as e:
            logger.error(
                f"{class_name}.should_enter [{time}]: IndexError - {str(e)}. "
                f"Likely insufficient data.",
                exc_info=True,
            )
            return False
        except AttributeError as e:
            logger.error(
                f"{class_name}.should_enter [{time}]: AttributeError - {str(e)}.",
                exc_info=True,
            )
            return False
        except TypeError as e:
            logger.error(
                f"{class_name}.should_enter [{time}]: TypeError - {str(e)}.",
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"{class_name}.should_enter [{time}]: Unexpected error: "
                f"{type(e).__name__} - {str(e)}",
                exc_info=True,
            )
            return False
