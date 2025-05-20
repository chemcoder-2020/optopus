import os
import pandas as pd
import dotenv
from loguru import logger
from .schwab_trade import SchwabTrade
from .schwab_data import SchwabData
from ...trades.option_spread import OptionStrategy
from ..order import Order
import time

dotenv.load_dotenv()


class SchwabOptionOrder(SchwabTrade, SchwabData, Order):
    def __init__(
        self,
        option_strategy: "OptionStrategy",
        client_id=os.getenv("SCHWAB_CLIENT_ID"),
        client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
        redirect_uri=os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1"),
        token_file=os.getenv("SCHWAB_TOKEN_FILE", "token.json"),
        which_account=0,
    ):
        # Initialize SchwabTrade and SchwabData with the token data
        SchwabTrade.__init__(
            self,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_file=token_file,
        )
        self.account_numbers = self.get_account_numbers()
        SchwabData.__init__(
            self,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_file=token_file,
            auth=self.auth,
        )

        self.account_number_hash_value = self.account_numbers[which_account][
            "hashValue"
        ]

        Order.__init__(self, option_strategy)

        self.order_status = None
        self.order_id = None
        self.exit_order_id = None

        self._broker = "Schwab"

    @property
    def broker(self):
        return self._broker

    def submit_entry(self):  # Removed price_step, wait_time
        # Ritual to avoid false price

        # 1. Fetch quotes
        symbols = [leg.schwab_symbol for leg in self.legs]
        new_option_chain_df = self.get_quote(f"{','.join(symbols)}")

        # 2. Update the current attributes of each leg
        current_time = new_option_chain_df["QUOTE_READTIME"].iloc[0]
        for leg in self.legs:
            leg.update(current_time, new_option_chain_df)

        # 3. Update the current price
        self.current_bid, self.current_ask = self.calculate_bid_ask()

        current_price = (self.current_bid + self.current_ask) / 2
        if self.strategy_side == "CREDIT":
            target_price = self.current_bid
        elif self.strategy_side == "DEBIT":
            target_price = self.current_ask
        else:
            raise ValueError("Invalid strategy side.")

        max_attempts = int(
            abs(target_price - current_price) // self.price_step
        ) + int(  # Use self.price_step
            abs(target_price - current_price) % self.price_step
            != 0  # Use self.price_step
        )

        for attempt in range(max_attempts):
            logger.info(
                f"Attempt {attempt + 1} to place entry order at price {current_price:.2f}"
            )
            payload = self.generate_entry_payload(current_price)
            result = super().place_order(self.account_number_hash_value, payload)
            if result:
                assert (
                    result[0] != "" and result[0] is not None
                ), "Order ID is empty when placing entry order."
                logger.info(f"Entry order placed successfully. Order ID: {result[0]}")
                time.sleep(self.wait_time)  # Use self.wait_time
                current_order = self.get_order(order_url=result[0])
                if current_order.get("status") == "FILLED":
                    logger.info("Entry order filled.")
                    self.order_id = result[0]
                    return result
                elif current_order.get("cancelable"):
                    logger.warning(
                        "Entry order not filled. Cancelling order and retrying."
                    )
                    if self.cancel_order(order_url=result[0]):
                        logger.info("Entry order canceled.")

                if target_price > current_price:
                    current_price += self.price_step  # Use self.price_step
                else:
                    current_price -= self.price_step  # Use self.price_step

            else:
                logger.warning(f"Entry attempt {attempt + 1} was not submitted.")
        logger.error("All attempts to place entry order failed.")
        return False

    def generate_entry_payload(self, price=None):
        if price is None:
            price = (self.current_bid + self.current_ask) / 2

        if self.strategy_type == "Vertical Spread":
            logger.info("Generating entry payload for vertical spread.")
            payload = self.generate_vertical_spread_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_option_type=self.legs[0].option_type[0],
                long_strike_price=self.legs[0].strike,
                short_option_type=self.legs[1].option_type[0],
                short_strike_price=self.legs[1].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=True,
            )
        elif self.strategy_type == "Iron Condor":
            logger.info("Generating entry payload for iron condor.")
            payload = self.generate_iron_condor_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_call_strike_price=self.legs[3].strike,
                short_call_strike_price=self.legs[2].strike,
                short_put_strike_price=self.legs[1].strike,
                long_put_strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=True,
            )
        elif self.strategy_type == "Iron Butterfly":
            logger.info("Generating entry payload for iron butterfly.")
            payload = self.generate_iron_butterfly_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_call_strike_price=self.legs[3].strike,
                strike_price=self.legs[2].strike,
                long_put_strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=True,
            )
        elif self.strategy_type == "Straddle":
            logger.info("Generating entry payload for straddle.")
            payload = self.generate_straddle_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=True,
            )
        elif self.strategy_type in ["Naked Put", "Naked Call"]:
            logger.info("Generating entry payload for naked option.")
            payload = self.generate_single_option_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                option_type=self.legs[0].option_type[0],
                strike_price=self.legs[0].strike,
                instruction=(
                    "BUY_TO_OPEN"
                    if self.legs[0].position_side == "BUY"
                    else "SELL_TO_OPEN"
                ),
                quantity=self.contracts,
                order_type="LIMIT",
                price=abs(price),
                duration="DAY",
            )
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        self._entry_payload = payload
        return payload

    def generate_exit_payload(self, price=None):
        if price is None:
            price = (self.current_bid + self.current_ask) / 2

        if self.strategy_type == "Vertical Spread":
            logger.info("Generating exit payload for vertical spread.")
            payload = self.generate_vertical_spread_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_option_type=self.legs[0].option_type[0],
                long_strike_price=self.legs[0].strike,
                short_option_type=self.legs[1].option_type[0],
                short_strike_price=self.legs[1].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=False,
            )
        elif self.strategy_type == "Iron Condor":
            logger.info("Generating exit payload for iron condor.")
            payload = self.generate_iron_condor_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_call_strike_price=self.legs[3].strike,
                short_call_strike_price=self.legs[2].strike,
                short_put_strike_price=self.legs[1].strike,
                long_put_strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=False,
            )
        elif self.strategy_type == "Iron Butterfly":
            logger.info("Generating exit payload for iron butterfly.")
            payload = self.generate_iron_butterfly_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                long_call_strike_price=self.legs[3].strike,
                strike_price=self.legs[2].strike,
                long_put_strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=False,
            )
        elif self.strategy_type == "Straddle":
            logger.info("Generating exit payload for straddle.")
            payload = self.generate_straddle_json(
                symbol=self.symbol,
                expiration=self.legs[0].expiration,
                strike_price=self.legs[0].strike,
                quantity=self.contracts,
                price=abs(price),
                duration="GOOD_TILL_CANCEL",
                is_entry=False,
            )
        elif self.strategy_type in ["Naked Put", "Naked Call"]:
            logger.info("Generating exit payload for naked option.")
            payload = self.generate_single_option_json(
                self.symbol,
                self.legs[0].expiration,
                self.legs[0].option_type[0],
                self.legs[0].strike,
                (
                    "BUY_TO_CLOSE"
                    if self.legs[0].position_side == "SELL"
                    else "SELL_TO_CLOSE"
                ),
                self.contracts,
                "LIMIT",
                abs(price),
                "GOOD_TILL_CANCEL",
            )
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        self._exit_payload = payload
        return payload

    def submit_exit(self):  # Removed price_step, wait_time

        # Ritual to avoid false price

        # 1. Fetch quotes
        symbols = [leg.schwab_symbol for leg in self.legs]
        new_option_chain_df = self.get_quote(f"{','.join(symbols)}")

        # 2. Update the current attributes of each leg
        current_time = new_option_chain_df["QUOTE_READTIME"].iloc[0]
        for leg in self.legs:
            leg.update(current_time, new_option_chain_df)

        # 3. Update the current price
        self.current_bid, self.current_ask = self.calculate_bid_ask()

        current_price = (self.current_bid + self.current_ask) / 2

        if self.strategy_type in ["Vertical Spread", "Iron Condor", "Iron Butterfly"]:
            target_price = self.current_ask
        else:
            target_price = self.current_bid

        # Guarding against false price

        # if (
        #     hasattr(self, "exit_scheme")
        #     and not hasattr(self.exit_scheme, "stoploss")
        #     and hasattr(self.exit_scheme, "profit_target")
        # ):  # if exit scheme has profit target but no stoploss
        #     if self.strategy_type in ["Vertical Spread", "Iron Condor", "Iron Butterfly"]:
        #         if current_price >= self.entry_net_premium:
        #             logger.info(
        #                 "Exit scheme doesn't have stoploss, only profit target. Current exit price exceeds entry price. Will not profit, so no exit. A false price has been detected. If status is CLOSED, will reopen strategy. Exiting..."
        #             )
        #             if self.status == "CLOSED":
        #                 self._reopen_strategy()  # reopen strategy
        #                 logger.info("Strategy was incorrectly closed. Now re-opened.")
        #             return False
        #     elif self.strategy_type in ["Naked Put", "Naked Call"]:
        #         if current_price <= self.entry_net_premium:
        #             logger.info(
        #                 "Exit scheme doesn't have stoploss, only profit target. Current exit price less than entry price. Will not profit, so no exit. A false price has been detected. If status is CLOSED, will reopen strategy. Exiting..."
        #             )
        #             if self.status == "CLOSED":
        #                 self._reopen_strategy()  # reopen strategy
        #                 logger.info("Strategy was incorrectly closed. Now re-opened.")
        #             return False

        max_attempts = int(
            abs(target_price - current_price) // self.price_step
        ) + int(  # Use self.price_step
            abs(target_price - current_price) % self.price_step
            != 0  # Use self.price_step
        )

        for attempt in range(max_attempts):
            logger.info(
                f"Attempt {attempt + 1} to place exit order at price {current_price:.2f}"
            )
            payload = self.generate_exit_payload(current_price)
            result = super().place_order(self.account_number_hash_value, payload)
            if result:
                assert (
                    result[0] != "" and result[0] is not None
                ), "Order ID is empty when placing exit order."
                logger.info(f"Exit order placed successfully. Order ID: {result[0]}")
                time.sleep(self.wait_time)  # Use self.wait_time
                current_order = self.get_order(order_url=result[0])
                if current_order.get("status") == "FILLED":
                    logger.info("Exit order filled.")
                    self.exit_order_id = result[0]
                    return result
                elif current_order.get("cancelable"):
                    logger.warning(
                        "Exit order not filled. Cancelling order and retrying."
                    )
                    if self.cancel_order(order_url=result[0]):
                        logger.info("Exit order canceled.")

                if target_price > current_price:
                    current_price += self.price_step  # Use self.price_step
                else:
                    current_price -= self.price_step  # Use self.price_step

            else:
                logger.warning(f"Exit attempt {attempt + 1} was not submitted.")
        logger.error("All attempts to place exit order failed.")
        return False

    def get_strategy_quote(self):
        symbols = [leg.schwab_symbol for leg in self.legs]
        new_option_chain_df = self.get_quote(f"{','.join(symbols)}")
        return new_option_chain_df

    def update_order(self, new_option_chain_df=None):
        if new_option_chain_df is None:
            # Fetch quotes
            new_option_chain_df = self.get_strategy_quote()

        self.current_time = new_option_chain_df["QUOTE_READTIME"].iloc[0]

        self.update(self.current_time, new_option_chain_df)
        if self.status == "CLOSED":
            for i in range(3):
                exit_submission_result = self.submit_exit()
                if exit_submission_result:
                    if self.exit_order_id:
                        exit_order = self.get_order(order_url=self.exit_order_id)
                        if exit_order:
                            self.exit_order_status = exit_order.get("status")
                            logger.info(
                                f"Order status updated to: {self.exit_order_status}"
                            )
                            if self.exit_order_status == "FILLED":
                                # Update exit price for each leg
                                order_df = self.parse_order(exit_order)
                                for leg in self.legs:
                                    leg_df = order_df.query(
                                        "`instrument.symbol` == @leg.schwab_symbol"
                                    )
                                    avg_leg_price = (
                                        leg_df["price"]
                                        * leg_df["quantity"]
                                        / leg_df["quantity"].sum()
                                    ).sum()
                                    leg.update_exit_price(avg_leg_price)

                                # Update exit net premium
                                self.update_exit_net_premium(order_response=exit_order)
                                break
        self.update_order_status()

    def update_entry_net_premium(self, order_response=None):
        """Update the entry net premium using data from a broker's order response.

        Args:
            order_response (dict): The order response from get_order() containing fill details
        """
        if order_response and order_response.get("status") == "FILLED":
            filled_price = abs(order_response.get("price", 0.0))
            self.entry_net_premium = filled_price
            logger.info(f"Updated from executed order: {self.entry_net_premium:.2f}")
        else:
            logger.warning("No order data provided - using calculated premium")
            self.entry_net_premium = sum(
                leg.entry_price * ratio * (1 if leg.position_side == "SELL" else -1)
                for leg, ratio in zip(self.legs, self.leg_ratios)
            )

    def update_exit_net_premium(self, order_response=None):
        """Update the exit net premium using data from a broker's order response.

        Args:
            order_response (dict): The order response from get_order() containing fill details
        """
        if order_response and order_response.get("status") == "FILLED":
            filled_price = abs(order_response.get("price", 0.0))
            self.exit_net_premium = filled_price
            logger.info(f"Updated from executed order: {self.exit_net_premium:.2f}")
        else:
            logger.warning("No order data provided - using calculated premium")
            self.exit_net_premium = sum(
                leg.entry_price * ratio * (1 if leg.position_side == "SELL" else -1)
                for leg, ratio in zip(self.legs, self.leg_ratios)
            )

    def update_order_status(self):
        if self.order_id:
            order = self.get_order(order_url=self.order_id)
            if order:
                previous_order_status = self.order_status
                self.order_status = order.get("status")
                if previous_order_status != self.order_status:
                    logger.info(f"Order status updated to: {self.order_status}")
                if self.order_status == "FILLED" and previous_order_status != "FILLED":
                    # Update entry price for each leg
                    order_df = self.parse_order(order)
                    for leg in self.legs:
                        leg_df = order_df.query(
                            "`instrument.symbol` == @leg.schwab_symbol"
                        )
                        avg_leg_price = (
                            leg_df["price"]
                            * leg_df["quantity"]
                            / leg_df["quantity"].sum()
                        ).sum()
                        leg.update_entry_price(avg_leg_price)

                    # Update entry net premium
                    self.update_entry_net_premium(order_response=order)

    def cancel(self):
        if self.order_id:
            result = self.cancel_order(order_url=self.order_id)
            if result:
                self.update_order_status()
                logger.info(
                    f"Order canceled successfully. New status: {self.order_status}"
                )
            else:
                logger.warning("Failed to cancel order.")
        else:
            logger.warning("No order ID available to cancel.")

    def modify(self, new_payload):
        if self.order_id:
            result = self.modify_order(order_url=self.order_id, payload=new_payload)
            if result:
                self.order_id = result[0]
                self.update_order_status()
                logger.info(
                    f"Order modified successfully. New status: {self.order_status}"
                )
            else:
                logger.warning("Failed to modify order.")
        else:
            logger.warning("No order ID available to modify.")

    def close_order(self):
        # Fetch quotes
        new_option_chain_df = self.get_strategy_quote()

        self.current_time = new_option_chain_df["QUOTE_READTIME"].iloc[0]

        self.update(self.current_time, new_option_chain_df)
        self.close_strategy(self.current_time, new_option_chain_df)

        for i in range(3):
            exit_submission_result = self.submit_exit()
            logger.info(f"Exit submission result: {exit_submission_result}")
            if exit_submission_result:
                if self.exit_order_id:
                    exit_order = self.get_order(order_url=self.exit_order_id)
                    if exit_order:
                        self.exit_order_status = exit_order.get("status")
                        logger.info(
                            f"Order status updated to: {self.exit_order_status}"
                        )
                        if self.exit_order_status == "FILLED":
                            # Update exit price for each leg
                            order_df = self.parse_order(exit_order)
                            for leg in self.legs:
                                leg_df = order_df.query(
                                    "`instrument.symbol` == @leg.schwab_symbol"
                                )
                                avg_leg_price = (
                                    leg_df["price"]
                                    * leg_df["quantity"]
                                    / leg_df["quantity"].sum()
                                ).sum()
                                leg.update_exit_price(avg_leg_price)

                            # Update exit net premium
                            self.update_exit_net_premium(order_response=exit_order)
                            break

    @staticmethod
    def parse_order(order_json):
        """Returns a DataFrame of the order details

        Args:
            order_json (dict): returned value from get_order()
        """
        leg_collection = pd.json_normalize(order_json["orderLegCollection"])
        activity_collection = pd.json_normalize(
            order_json["orderActivityCollection"], "executionLegs"
        )

        merged_df = pd.merge(
            leg_collection,
            activity_collection,
            on=["legId"],
            how="inner",
            suffixes=["_leg", "_activity"],
        )
        common_cols_all = leg_collection.columns.intersection(
            activity_collection.columns
        ).tolist()
        common_cols = [col for col in common_cols_all if col not in ["legId"]]
        for col in common_cols:
            merged_col_leg = f"{col}_leg"
            merged_col_activity = f"{col}_activity"

            merged_df[col] = merged_df[merged_col_leg].fillna(
                merged_df[merged_col_activity]
            )
            # Drop the original merged columns
            merged_df.drop([merged_col_leg, merged_col_activity], axis=1, inplace=True)

        return merged_df

    def __repr__(self):
        """
        Return a string representation combining both order details and strategy details.
        """
        legs_repr = "\n    ".join(repr(leg) for leg in self.legs)
        return (
            f"SchwabOptionOrder(\n"
            f"  Order ID: {self.order_id},\n"
            f"  Order Status: {self.order_status},\n"
            f"  Strategy Details:\n"
            f"    Symbol: {self.symbol},\n"
            f"    Strategy Type: {self.strategy_type},\n"
            f"    Status: {self.status},\n"
            f"    Contracts: {self.contracts},\n"
            f"    Entry Time: {self.entry_time},\n"
            f"    Current Time: {self.current_time},\n"
            f"    Entry Net Premium: {self.entry_net_premium:.2f},\n"
            f"    Current Net Premium: {self.net_premium:.2f},\n"
            f"    Return Percentage: {self.return_percentage():.2f}%,\n"
            f"    Current Delta: {self.current_delta():.2f},\n"
            f"    Days in Trade (DIT): {self.DIT},\n"
            f"    Strategy Legs:\n    {legs_repr}\n"
            f")"
        )


# if __name__ == "__main__":
#     # Load sample option chain data
#     entry_df = pd.read_parquet(
#         "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-26 15-15.parquet"
#     )

#     # Create a vertical spread strategy
#     vertical_spread = OptionStrategy.create_vertical_spread(
#         symbol="SPY",
#         option_type="PUT",
#         long_strike="-2",
#         short_strike="ATM",
#         expiration="2024-11-15",
#         contracts=1,
#         entry_time="2024-09-26 15:15:00",
#         option_chain_df=entry_df,
#     )

#     # Create a SchwabOptionOrder instance

#     schwab_order = SchwabOptionOrder(
#         client_id=os.getenv("SCHWAB_CLIENT_ID"),
#         client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
#         option_strategy=vertical_spread,
#         token_file="token.json",
#     )

#     # Print order details
#     print("Schwab Option Order Details:")
#     print(f"Symbol: {schwab_order.symbol}")
#     print(f"Strategy Type: {schwab_order.strategy_type}")
#     print(f"Expiration: {schwab_order.legs[0].expiration}")
#     print(f"Long Strike: {schwab_order.legs[0].strike}")
#     print(f"Short Strike: {schwab_order.legs[1].strike}")
#     print(f"Contracts: {schwab_order.contracts}")
#     print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
#     print(f"Order Status: {schwab_order.order_status}")
#     print(f"Time: {schwab_order.current_time}")
#     print(f"DIT: {schwab_order.DIT}")
#     print(f"Net Premium: {schwab_order.net_premium}")

#     schwab_order.update_order()

#     print("Schwab Option Order Details (After Update):")
#     print(f"Symbol: {schwab_order.symbol}")
#     print(f"Strategy Type: {schwab_order.strategy_type}")
#     print(f"Expiration: {schwab_order.legs[0].expiration}")
#     print(f"Long Strike: {schwab_order.legs[0].strike}")
#     print(f"Short Strike: {schwab_order.legs[1].strike}")
#     print(f"Contracts: {schwab_order.contracts}")
#     print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
#     print(f"Order Status: {schwab_order.order_status}")
#     print(f"Time: {schwab_order.current_time}")
#     print(f"DIT: {schwab_order.DIT}")
#     print(f"Net Premium: {schwab_order.net_premium}")
