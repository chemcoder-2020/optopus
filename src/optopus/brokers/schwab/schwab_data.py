from loguru import logger
import pandas as pd
import os
from .schwab import Schwab



class SchwabData(Schwab):
    def __init__(
        self,
        client_id,
        client_secret,
        redirect_uri="https://127.0.0.1",
        token_file="token.json",
        auth=None,
    ):
        super().__init__(client_id, client_secret, redirect_uri, token_file, auth)
        self.marketdata_base_url = "https://api.schwabapi.com/marketdata/v1"

    def get_option_chain(
        self,
        symbol,
        contract_type="ALL",
        strike_count=160,
        include_underlying_quote=None,
        strategy="SINGLE",
        interval=None,
        strike=None,
        range=None,
        from_date=None,
        to_date=None,
        volatility=None,
        underlying_price=None,
        interest_rate=None,
        days_to_expiration=None,
        exp_month=None,
        option_type=None,
        entitlement=None,
        output_folders=None,
    ):
        """
        Get Option Chain including information on options contracts associated with each expiration.

        Parameters:
            symbol (str): The underlying asset symbol.
            contract_type (str): Contract Type. Available values: CALL, PUT, ALL. Default: ALL.
            strike_count (int): The Number of strikes to return above or below the at-the-money price.
            include_underlying_quote (bool): Underlying quotes to be included.
            strategy (str): OptionChain strategy. Default is SINGLE. Available values: SINGLE, ANALYTICAL, COVERED, VERTICAL, CALENDAR, STRANGLE, STRADDLE, BUTTERFLY, CONDOR, DIAGONAL, COLLAR, ROLL.
            interval (float): Strike interval for spread strategy chains (see strategy param).
            strike (float): Strike Price.
            range (str): Range (ITM/NTM/OTM etc.).
            from_date (str): From date (pattern: yyyy-MM-dd).
            to_date (str): To date (pattern: yyyy-MM-dd).
            volatility (float): Volatility to use in calculations. Applies only to ANALYTICAL strategy chains.
            underlying_price (float): Underlying price to use in calculations. Applies only to ANALYTICAL strategy chains.
            interest_rate (float): Interest rate to use in calculations. Applies only to ANALYTICAL strategy chains.
            days_to_expiration (int): Days to expiration to use in calculations. Applies only to ANALYTICAL strategy chains.
            exp_month (str): Expiration month. Available values: JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC, ALL.
            option_type (str): Option Type.
            entitlement (str): Applicable only if its retail token, entitlement of client. Available values: PN, NP, PP.
        """
        url = f"{self.marketdata_base_url}/chains"
        params = {
            "symbol": symbol,
            "contractType": contract_type,
            "strikeCount": strike_count,
            "includeUnderlyingQuote": include_underlying_quote,
            "strategy": strategy,
            "interval": interval,
            "strike": strike,
            "range": range,
            "fromDate": from_date,
            "toDate": to_date,
            "volatility": volatility,
            "underlyingPrice": underlying_price,
            "interestRate": interest_rate,
            "daysToExpiration": days_to_expiration,
            "expMonth": exp_month,
            "optionType": option_type,
            "entitlement": entitlement,
        }
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        response = self._get(url, params=params)
        response.raise_for_status()
        option_chain = response.json()
        processed_chain = self._process_option_chain(option_chain)

        if output_folders:
            timestamp = (
                pd.Timestamp.now(tz="America/New_York")
                .round("15min")
                .strftime("%Y-%m-%d %H-%M")
            )
            filename = f"{symbol}_{timestamp}.parquet"
            if not isinstance(output_folders, list):
                output_folders = [output_folders]
            for output_folder in output_folders:
                filepath = os.path.join(output_folder, filename)
                processed_chain.to_parquet(filepath)

        return processed_chain

    def get_market_hours(self, market_id, date=None):
        """
        Get Market Hours for dates in the future for a single market.

        Parameters:
            market_id (str): Market ID. Available values: equity, option, bond, future, forex.
            date (str, optional): Valid date range is from current date to 1 year from today. It will default to current day if not entered. Date format: YYYY-MM-DD.
        """
        url = f"{self.marketdata_base_url}/markets"
        params = {"date": date, "markets": market_id}
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        response = self._get(url, params=params)
        response.raise_for_status()
        return response.json()

    def market_isOpen(self, market_id="option", date=None):
        """
        Check if the market is open for a given market and date.

        Parameters:
            market_id (str): Market ID. Available values: equity, option, bond, future, forex. Future has many keys, so it's not supported in this method. Refer to get_market_hours for a list of markets.
            date (str, optional): Valid date range is from current date to 1 year from today. It will default to current day if not entered. Date format: YYYY-MM-DD.

        Returns:
            bool: True if the market is open, False otherwise.
        """
        market_hours = self.get_market_hours(market_id, date)
        if market_id == "option":
            last_key = "EQO"
        elif market_id == "equity":
            last_key = "EQ"
        elif market_id == "forex":
            last_key = "forex"
        elif market_id == "bond":
            last_key = "BON"
        else:
            raise ValueError(f"Invalid market_id: {market_id}")
        return market_hours[market_id][last_key]["isOpen"]

    def get_quote(self, symbols, fields="all", indicative=False):
        """
        Get quote for a symbol or list of symbols.

        Parameters:
            symbols (str): Comma separated list of symbol(s) to look up a quote.
            fields (str, optional): Request for subset of data by passing comma separated list of root nodes. Possible root nodes are quote, fundamental, extended, reference, regular. Default value: all.
            indicative (bool, optional): Include indicative symbol quotes for all ETF symbols in request. Default value: false.

        Returns:
            dict: The response data.
        """
        url = f"{self.marketdata_base_url}/quotes"
        params = {
            "symbols": symbols,
            "fields": fields,
            "indicative": str(indicative).lower(),
        }
        response = self._get(url, params=params)
        response.raise_for_status()
        raw_quote = response.json()
        return self.format_quote(raw_quote)

    def format_quote(self, raw_quote):
        """
        Format the output of get_quote to match the structure and data types of the output from _process_option_chain.

        Parameters:
            raw_quote (dict): The raw quote data from the API response.

        Returns:
            pd.DataFrame: The formatted quote data.
        """
        formatted_quotes = []
        for symbol, data in raw_quote.items():
            quote = data["quote"]
            reference = data["reference"]
            contractType = reference["contractType"]
            formatted_quote = {
                f"{contractType}_BID": quote.get("bidPrice", 0.0),
                f"{contractType}_ASK": quote.get("askPrice", 0.0),
                f"{contractType}_LAST": quote.get("lastPrice", 0.0),
                f"{contractType}_MARK": quote.get("mark", 0.0),
                f"{contractType}_DELTA": quote.get("delta", 0.0),
                "STRIKE": reference.get("strikePrice", 0.0),
                "EXPIRE_DATE": pd.to_datetime(
                    f"{reference['expirationYear']}-{reference['expirationMonth']}-{reference['expirationDay']}"
                ),
                "DTE": reference.get("daysToExpiration", 0.0),
                f"{contractType}_ITM": quote.get("moneyIntrinsicValue", 0.0) > 0,
                "UNDERLYING_LAST": quote.get("underlyingPrice", 0.0),
                "QUOTE_READTIME": pd.to_datetime(quote.get("quoteTime", 0), unit="ms")
                .tz_localize("UTC")
                .tz_convert("America/New_York")
                .tz_localize(None),
                "QUOTE_TIME_HOURS": pd.to_datetime(quote.get("quoteTime", 0), unit="ms")
                .tz_localize("UTC")
                .tz_convert("America/New_York")
                .tz_localize(None)
                .hour
                + pd.to_datetime(quote.get("quoteTime", 0), unit="ms")
                .tz_localize("UTC")
                .tz_convert("America/New_York")
                .tz_localize(None)
                .minute
                / 60,
            }
            # Calculate MARK if not available and set LAST to MARK if LAST is not available
            formatted_quote[f"{contractType}_MARK"] = (
                formatted_quote[f"{contractType}_MARK"]
                if formatted_quote[f"{contractType}_MARK"]
                else (
                    formatted_quote[f"{contractType}_BID"]
                    + formatted_quote[f"{contractType}_ASK"]
                )
                / 2
            )
            formatted_quote[f"{contractType}_LAST"] = (
                formatted_quote[f"{contractType}_LAST"]
                if formatted_quote[f"{contractType}_LAST"]
                else formatted_quote[f"{contractType}_MARK"]
            )
            formatted_quote["intDTE"] = int(formatted_quote["DTE"])
            formatted_quotes.append(formatted_quote)

        formatted_df = pd.DataFrame(formatted_quotes)
        formatted_df = formatted_df[
            [
                f"{contractType}_BID",
                f"{contractType}_ASK",
                f"{contractType}_LAST",
                f"{contractType}_MARK",
                f"{contractType}_DELTA",
                "STRIKE",
                "EXPIRE_DATE",
                "DTE",
                f"{contractType}_ITM",
                "UNDERLYING_LAST",
                "QUOTE_READTIME",
                "QUOTE_TIME_HOURS",
                "intDTE",
            ]
        ]
        formatted_df[f"{contractType}_ITM"] = formatted_df[
            f"{contractType}_ITM"
        ].astype(bool)
        formatted_df["intDTE"] = formatted_df["intDTE"].astype(int)
        formatted_df["QUOTE_READTIME"] = pd.to_datetime(
            formatted_df["QUOTE_READTIME"]
        ).round("15min")
        for col in [
            f"{contractType}_BID",
            f"{contractType}_ASK",
            f"{contractType}_LAST",
            f"{contractType}_MARK",
            f"{contractType}_DELTA",
            "STRIKE",
            "DTE",
            "UNDERLYING_LAST",
        ]:
            formatted_df[col] = formatted_df[col].astype("float64")
        return formatted_df

    def process_price_history(self, price_history_json):
        """
        Process the price history JSON response and return a DataFrame.

        Parameters:
            price_history_json (dict): The JSON response from the price history API.

        Returns:
            pd.DataFrame: DataFrame with columns open, high, low, close, volume, datetime.
        """
        df = pd.DataFrame(price_history_json["candles"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
        df = df[["open", "high", "low", "close", "volume", "datetime"]]
        return df

    def get_price_history(
        self,
        symbol,
        period_type="day",
        period=None,
        frequency_type="minute",
        frequency=1,
        start_date=None,
        end_date=None,
        need_extended_hours_data=False,
        need_previous_close=False,
    ):
        """
        Get historical Open, High, Low, Close, and Volume for a given frequency.

        Parameters:
            symbol (str): The Equity symbol used to look up price history.
            period_type (str): The chart period being requested. Valid values: day, month, year, ytd.
            period (int): The number of chart period types.
            frequency_type (str): The time frequency type. Valid values depend on period_type.
            frequency (int): The time frequency duration. Valid values depend on frequency_type.
            start_date (int): The start date, Time in milliseconds since the UNIX epoch.
            end_date (int): The end date, Time in milliseconds since the UNIX epoch.
            need_extended_hours_data (bool): Need extended hours data.
            need_previous_close (bool): Need previous close price/date.
        """
        url = f"{self.marketdata_base_url}/pricehistory"
        params = {
            "symbol": symbol,
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
            "startDate": start_date,
            "endDate": end_date,
            "needExtendedHoursData": str(need_extended_hours_data).lower(),
            "needPreviousClose": str(need_previous_close).lower(),
        }
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        response = self._get(url, params=params)
        response.raise_for_status()
        price_history_json = response.json()
        return self.process_price_history(price_history_json)

    @classmethod
    def _process_option_chain(cls, opt_chain):
        calls = opt_chain["callExpDateMap"]
        puts = opt_chain["putExpDateMap"]
        chain = []
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
        chain_calls = pd.concat(
            [pd.DataFrame(v[k2][0])[info] for k, v in calls.items() for k2 in v],
            ignore_index=True,
        )
        chain_puts = pd.concat(
            [pd.DataFrame(v[k2][0])[info] for k, v in puts.items() for k2 in v],
            ignore_index=True,
        )
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
        chain_calls["MARK"] = chain_calls["MARK"].combine_first(
            (chain_calls["BID"] + chain_calls["ASK"]) / 2
        )
        chain_calls["LAST"] = chain_calls["LAST"].combine_first(chain_calls["MARK"])
        chain_puts["MARK"] = chain_puts["MARK"].combine_first(
            (chain_puts["BID"] + chain_puts["ASK"]) / 2
        )
        chain_puts["LAST"] = chain_puts["LAST"].combine_first(chain_puts["MARK"])

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
        chain["EXPIRE_DATE"] = pd.to_datetime(chain["EXPIRE_DATE"].dt.date)

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
        return chain
