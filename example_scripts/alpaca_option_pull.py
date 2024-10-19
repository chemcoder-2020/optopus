import pandas as pd
import re
from alpaca.data import OptionChainRequest, OptionHistoricalDataClient, StockHistoricalDataClient, StockLatestBarRequest
import json
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
bot_params = json.load(open(f"{cwd}/tradier_params.json"))
ticker = bot_params['ticker']
r = bot_params['r']

api_key = bot_params['alpaca_api_key']
secret_key = bot_params['alpaca_secret_key']
def parse_option_symbol(symbol):
    underlying, expire_date, option_type, strike = re.match(
        r"(.{2,3})(\d{6})([CP])(\d{8})", symbol
    ).groups()
    expire_date = pd.Timestamp(
        "20" + expire_date[:2] + "-" + expire_date[2:4] + "-" + expire_date[4:]
    )
    strike = int(strike) / 1000.0
    return {"EXPIRE_DATE": expire_date, "OPTION_TYPE": option_type, "STRIKE": strike}


client = OptionHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key,
)
stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

request_params = OptionChainRequest(
    underlying_symbol=ticker,
    # feed='opra'
)
option_chain = client.get_option_chain(request_params)

stock_request = StockLatestBarRequest(symbol_or_symbols=ticker)
stock_quotes = stock_client.get_stock_latest_bar(stock_request)

puts = []
calls = []
for symbol, option in option_chain.items():
    symbol_parse = parse_option_symbol(symbol)
    option_type = symbol_parse["OPTION_TYPE"]

    row = {
        f"{option_type}_ASK": option.latest_quote.ask_price,
        f"{option_type}_BID": option.latest_quote.bid_price,
        "QUOTE_READTIME": pd.Timestamp(option.latest_quote.timestamp)
        .tz_convert("US/Eastern").tz_localize(None)
        .round("15min"),
        f"{option_type}_IV": option.implied_volatility,
        f"{option_type}_LAST": option.latest_trade.price if option.latest_trade is not None else (option.latest_quote.ask_price + option.latest_quote.bid_price) / 2,
        f"{option_type}_MARK": (option.latest_quote.ask_price + option.latest_quote.bid_price) / 2,
    }

    row.update(symbol_parse)

    if option_type == "C":
        row[f'{option_type}_ITM'] = row['STRIKE'] - stock_quotes[ticker].close < 0
    else:
        row[f'{option_type}_ITM'] = row['STRIKE'] - stock_quotes[ticker].close > 0
    if option.greeks is None:
        row.update(
            {
                f"{option_type}_DELTA": pd.NA,
                f"{option_type}_GAMMA": pd.NA,
                f"{option_type}_THETA": pd.NA,
                f"{option_type}_VEGA": pd.NA,
                f"{option_type}_RHO": pd.NA,
            }
        )
    else:
        row.update(
            {
                f"{option_type}_DELTA": (
                    option.greeks.delta if option.greeks.delta is not None else pd.NA
                ),
                f"{option_type}_GAMMA": (
                    option.greeks.gamma if option.greeks.gamma is not None else pd.NA
                ),
                f"{option_type}_THETA": (
                    option.greeks.theta if option.greeks.theta is not None else pd.NA
                ),
                f"{option_type}_VEGA": (
                    option.greeks.vega if option.greeks.vega is not None else pd.NA
                ),
                f"{option_type}_RHO": (
                    option.greeks.rho if option.greeks.rho is not None else pd.NA
                ),
            }
        )
    row.pop("OPTION_TYPE")
    
    # row['EXPIRE_DATE']
    # row['QUOTE_READTIME'].date()

    # (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days + ((16 -df['QUOTE_TIME_HOURS'])/24).round(4)

    if option_type == "P":
        puts.append(row)
    else:
        calls.append(row)

puts = pd.DataFrame(puts)
calls = pd.DataFrame(calls)

now = pd.Timestamp.now(tz="America/New_York")
df = puts.merge(calls, on=["EXPIRE_DATE", "STRIKE"], how="inner")
df["QUOTE_READTIME"] = now.round("1min")
df.drop(['QUOTE_READTIME_x', 'QUOTE_READTIME_y'], axis=1, inplace=True)

df['QUOTE_TIME_HOURS'] = df['QUOTE_READTIME'].dt.hour + df['QUOTE_READTIME'].dt.minute / 60
df['DTE'] = (df['EXPIRE_DATE'] - pd.DatetimeIndex(df['QUOTE_READTIME'].dt.date)).dt.days + ((16 -df['QUOTE_TIME_HOURS'])/24).round(4)
df['intDTE'] = df['DTE'].astype(int)
df['UNDERLYING_LAST'] = stock_quotes[ticker].close
df['INTEREST_RATE'] = r


year = now.year
schwab_chain_dir = "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains"


if not os.path.exists(os.path.join(schwab_chain_dir, ticker)):
    os.mkdir(os.path.join(schwab_chain_dir, ticker))

if not os.path.exists(os.path.join(schwab_chain_dir, ticker, str(year))):
    os.mkdir(os.path.join(schwab_chain_dir, ticker, str(year)))

print(f"{now.floor("1min").strftime("%Y-%m-%d %H-%M")} Pulling option chain for {ticker}")

df.to_parquet(
    os.path.join(
        schwab_chain_dir,
        ticker,
        str(year),
        ticker + "_" + now.floor("1min").strftime("%Y-%m-%d %H-%M") + ".parquet",
    )
)
