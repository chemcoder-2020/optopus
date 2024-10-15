import os
import sys
import pandas as pd
from dotenv import load_dotenv, dotenv_values
from loguru import logger

from .schwab_trade import SchwabTrade
from .schwab_data import SchwabData
from ..trades.option_spread import OptionStrategy
from ..order import Order

logger.add(
    sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO"
)

dotenv.load_dotenv()


class SchwabOptionOrder(SchwabTrade, SchwabData, Order):
    def __init__(
        self,
        option_strategy: OptionStrategy,
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
        OptionStrategy.__init__(
            self,
            option_strategy.symbol,
            option_strategy.strategy_type,
            option_strategy.profit_target,
            option_strategy.stop_loss,
            option_strategy.trailing_stop,
            option_strategy.contracts,
            option_strategy.commission,
        )
        Order.__init__(self)

        self.order_status = None
        self.order_id = None

        self._broker = "Schwab"

    @property
    def broker(self):
        return self._broker

    def generate_entry_payload(self):
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
                price=abs(self.current_bid),
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
                price=abs(self.entry_net_premium),
                duration="DAY",
            )
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        self._entry_payload = payload
        return payload

    def submit_entry(self):
        self.update_order()  # update fresh quotes
        payload = self.generate_entry_payload()
        result = super().place_order(self.account_number_hash_value, payload)
        if result:
            self.order_id = result[0]
            assert (
                self.order_id != "" and self.order_id is not None
            ), "Order ID is empty when placing order."
            self.update_order_status()
        return result

    def generate_exit_payload(self):
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
                price=abs(self.current_ask),
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
                abs(self.net_premium),
                "GOOD_TILL_CANCEL",
            )
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        self._exit_payload = payload
        return payload

    def submit_exit(self):
        self.update_order()  # update fresh quotes
        payload = self.generate_exit_payload()
        result = super().place_order(self.account_number_hash_value, payload)
        if result:
            self.order_id = result[0]
            assert (
                self.order_id != "" and self.order_id is not None
            ), "Order ID is empty when placing order."
            self.update_order_status()
        return result

    def update_order(self, new_option_chain_df=None):
        if new_option_chain_df is None:
            # Fetch quotes
            symbols = [leg.schwab_symbol for leg in self.legs]
            new_option_chain_df = self.get_quote(f"{','.join(symbols)}")

        self.current_time = new_option_chain_df["QUOTE_READTIME"].iloc[0]

        self.update(self.current_time, new_option_chain_df)
        if self.status == "CLOSED":
            self.submit_exit()
        self.update_order_status()

    def update_order_status(self):
        if self.order_id:
            order = self.get_order(order_url=self.order_id)
            if order:
                self.order_status = order.get("status")
                logger.info(f"Order status updated to: {self.order_status}")
                if self.order_status == "FILLED":
                    # Update entry price for each leg
                    for leg_num, leg in enumerate(self.legs):
                        leg.update_entry_price(
                            order["orderActivityCollection"][0]["executionLegs"][
                                leg_num
                            ]["price"]
                        )

                    # Update entry net premium
                    self.update_entry_net_premium()

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

    def __repr__(self):
        return (
            f"SchwabOptionOrder(\n"
            f"  Order ID: {self.order_id},\n"
            f"  Order Status: {self.order_status}\n"
            f")"
        )


if __name__ == "__main__":
    # Load sample option chain data
    entry_df = pd.read_parquet(
        "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains/SPY/2024/SPY_2024-09-26 15-15.parquet"
    )

    # Create a vertical spread strategy
    vertical_spread = OptionStrategy.create_vertical_spread(
        symbol="SPY",
        option_type="PUT",
        long_strike="-2",
        short_strike="ATM",
        expiration="2024-11-15",
        contracts=1,
        entry_time="2024-09-26 15:15:00",
        option_chain_df=entry_df,
    )

    # Create a SchwabOptionOrder instance

    schwab_order = SchwabOptionOrder(
        client_id=os.getenv("SCHWAB_CLIENT_ID"),
        client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
        option_strategy=vertical_spread,
        token_file="token.json",
    )

    # Print order details
    print("Schwab Option Order Details:")
    print(f"Symbol: {schwab_order.symbol}")
    print(f"Strategy Type: {schwab_order.strategy_type}")
    print(f"Expiration: {schwab_order.legs[0].expiration}")
    print(f"Long Strike: {schwab_order.legs[0].strike}")
    print(f"Short Strike: {schwab_order.legs[1].strike}")
    print(f"Contracts: {schwab_order.contracts}")
    print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
    print(f"Order Status: {schwab_order.order_status}")
    print(f"Time: {schwab_order.current_time}")
    print(f"DIT: {schwab_order.DIT}")
    print(f"Net Premium: {schwab_order.net_premium}")

    schwab_order.update_order()

    print("Schwab Option Order Details (After Update):")
    print(f"Symbol: {schwab_order.symbol}")
    print(f"Strategy Type: {schwab_order.strategy_type}")
    print(f"Expiration: {schwab_order.legs[0].expiration}")
    print(f"Long Strike: {schwab_order.legs[0].strike}")
    print(f"Short Strike: {schwab_order.legs[1].strike}")
    print(f"Contracts: {schwab_order.contracts}")
    print(f"Entry Net Premium: {schwab_order.entry_net_premium:.2f}")
    print(f"Order Status: {schwab_order.order_status}")
    print(f"Time: {schwab_order.current_time}")
    print(f"DIT: {schwab_order.DIT}")
    print(f"Net Premium: {schwab_order.net_premium}")

    # Example of placing the order (Note: This won't actually execute without valid credentials)

    # schwab_order.submit_entry()
    # schwab_order.submit_exit()

    # Example of updating the order
    # update_df = pd.read_parquet("path/to/your/updated_option_chain_data.parquet")
    # schwab_order.update_order(update_df)
    # print(f"Updated Order Status: {schwab_order.order_status}")
    # print(f"Current Net Premium: {schwab_order.net_premium:.2f}")
