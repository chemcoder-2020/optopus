from abc import ABC, abstractmethod
import schwab
import pandas as pd
import sys


class OptionDataAPI(ABC):
    @classmethod
    def from_client_name(cls, client_name, *args, **kwargs):
        if client_name.lower() == "schwab":
            return SchwabOptionDataAPI(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported client: {client_name}")

    def get_option_chain(self, symbol, *args, **kwargs):
        return self._get_option_chain(symbol, *args, **kwargs)

    @abstractmethod
    def _get_option_chain(self, symbol, *args, **kwargs):
        pass


class SchwabOptionDataAPI(OptionDataAPI):
    def __init__(self, api_key, api_secret, token_filepath):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token_filepath = token_filepath
        self.client = schwab.auth.client_from_token_file(
            self.token_filepath,
            api_key=self.api_key,
            app_secret=self.api_secret,
        )

    @classmethod
    def _process_option_chain(cls, opt_chain):
        calls = opt_chain["callExpDateMap"]
        puts = opt_chain["putExpDateMap"]
        chain = []
        chain_calls = []
        chain_puts = []
        for k, v in calls.items():
            info = [
                "bid",
                "ask",
                "last",
                "mark",
                "delta",
                "strikePrice",
                "expirationDate",
                "daysToExpiration",
                "inTheMoney",
            ]
            for k2, v2 in v.items():
                df = pd.DataFrame(v[k2][0])[info]
                chain_calls.append(df)
                chain.append(df)

        for k, v in puts.items():
            info = [
                "bid",
                "ask",
                "last",
                "mark",
                "delta",
                "strikePrice",
                "expirationDate",
                "daysToExpiration",
                "inTheMoney",
            ]
            for k2, v2 in v.items():
                df = pd.DataFrame(v[k2][0])[info]
                chain_puts.append(df)
                chain.append(df)

        chain_calls = pd.concat(chain_calls, ignore_index=True)
        chain_puts = pd.concat(chain_puts, ignore_index=True)
        rename = {
            "bid": "BID",
            "ask": "ASK",
            "last": "LAST",
            "mark": "MARK",
            "delta": "DELTA",
            "strikePrice": "STRIKE",
            "expirationDate": "EXPIRE_DATE",
            "daysToExpiration": "DTE",
            "inTheMoney": "ITM",
        }
        chain_calls.rename(columns=rename, inplace=True)
        chain_puts.rename(columns=rename, inplace=True)

        # Calculate MARK if not available and set LAST to MARK if LAST is not available
        chain_calls["MARK"] = chain_calls["MARK"].fillna(
            (chain_calls["BID"] + chain_calls["ASK"]) / 2
        )
        chain_calls["LAST"] = chain_calls["LAST"].fillna(chain_calls["MARK"])
        chain_puts["MARK"] = chain_puts["MARK"].fillna(
            (chain_puts["BID"] + chain_puts["ASK"]) / 2
        )
        chain_puts["LAST"] = chain_puts["LAST"].fillna(chain_puts["MARK"])

        chain = pd.merge(
            chain_calls,
            chain_puts,
            how="inner",
            on=["STRIKE", "EXPIRE_DATE", "DTE"],
            suffixes=("_C", "_P"),
        )
        suffixed_cols = [
            col for col in chain.columns if col[-2:] == "_C" or col[-2:] == "_P"
        ]
        prefixed_cols = [
            col.split("_")[-1] + "_" + col.split("_")[0] for col in suffixed_cols
        ]
        chain.rename(columns=dict(zip(suffixed_cols, prefixed_cols)), inplace=True)
        chain["intDTE"] = chain["DTE"].astype(int)
        chain["UNDERLYING_LAST"] = opt_chain["underlyingPrice"]
        chain["INTEREST_RATE"] = opt_chain["interestRate"]
        now = pd.Timestamp.now(tz="America/New_York").round("15min").tz_localize(None)
        chain["QUOTE_READTIME"] = now
        chain["QUOTE_TIME_HOURS"] = now.hour + now.minute / 60
        chain["EXPIRE_DATE"] = (
            pd.to_datetime(chain["EXPIRE_DATE"])
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

        dte_diff = (chain["EXPIRE_DATE"] - now).dt
        chain["DTE"] = (
            dte_diff.days
            + dte_diff.components.hours / 24
            + dte_diff.components.minutes / 60
        ).round(4)
        chain["EXPIRE_DATE"] = pd.to_datetime(
            pd.to_datetime(chain["EXPIRE_DATE"]).dt.date
        )

        # Select only the required columns
        required_cols = [
            "C_BID",
            "C_ASK",
            "C_LAST",
            "C_MARK",
            "C_DELTA",
            "STRIKE",
            "EXPIRE_DATE",
            "DTE",
            "C_ITM",
            "P_BID",
            "P_ASK",
            "P_LAST",
            "P_MARK",
            "P_DELTA",
            "P_ITM",
            "intDTE",
            "UNDERLYING_LAST",
            "QUOTE_READTIME",
        ]
        chain = chain[required_cols]

        # Convert columns to specified data types
        chain["C_ITM"] = chain["C_ITM"].astype(bool)
        chain["P_ITM"] = chain["P_ITM"].astype(bool)
        chain["intDTE"] = chain["intDTE"].astype(int)
        chain["QUOTE_READTIME"] = pd.to_datetime(chain["QUOTE_READTIME"])
        for col in [
            "C_BID",
            "C_ASK",
            "C_LAST",
            "C_MARK",
            "C_DELTA",
            "STRIKE",
            "DTE",
            "P_BID",
            "P_ASK",
            "P_LAST",
            "P_MARK",
            "P_DELTA",
            "UNDERLYING_LAST",
        ]:
            chain[col] = chain[col].astype("float64")
        # chain
        return chain

    def _get_option_chain(
        self, symbol, strike_count=50, include_underlying_quote=False
    ):
        opt_chain = self.client.get_option_chain(
            symbol,
            include_underlying_quote=include_underlying_quote,
            strike_count=strike_count,
        )
        opt_chain = opt_chain.json()
        return self._process_option_chain(opt_chain)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python option_data_api.py <client_name> <api_key> <api_secret> <token_filepath>"
        )
        sys.exit(1)

    client_name = sys.argv[1]
    api_key = sys.argv[2]
    api_secret = sys.argv[3]
    token_filepath = sys.argv[4]

    option_api = OptionDataAPI.from_client_name(
        client_name, api_key, api_secret, token_filepath
    )
    option_chain = option_api.get_option_chain("SPY")
    print(option_chain.sort_values(by=["DTE"]))
