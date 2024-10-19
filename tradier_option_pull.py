import requests
from py_vollib.black_scholes_merton.greeks.analytical import delta
import pandas as pd
import os
from joblib import Parallel, delayed
import json



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
bot_params = json.load(open(f"{cwd}/tradier_params.json"))
ticker = bot_params['ticker']
r = bot_params['r']
div = bot_params['div']
token = bot_params['tradier_token']
base_url = bot_params['tradier_base_url']
max_dte = bot_params['SPS']['DTE_max']

now = pd.Timestamp.now()
now = now.tz_localize("US/Pacific").tz_convert("US/Eastern").tz_localize(None)
print("-----------------------------------")
print(f"{now.round("15min")} - Fetching option chain for {ticker}...")
date = now.date()
future_dates = pd.date_range(date, periods=max_dte, freq="B")
future_dates = future_dates.strftime("%Y-%m-%d").tolist()

# underlying quote
response = requests.get(
    f"{base_url}/v1/markets/quotes",
    params={"symbols": ticker},
    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
)
json_response = response.json()
underlying_last = json_response['quotes']['quote']['last']


def fetch_option_chain(future_date):
    if now.day_of_week < 5:
        try:
            response = requests.get(
                f"{base_url}/v1/markets/options/chains",
                params={"symbol": ticker, "expiration": future_date, "greeks": "true"},
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )

            json_response = response.json()
            # print(response.status_code)
            df = pd.DataFrame(json_response["options"]["option"])
            df = df.join(df.pop("greeks").apply(pd.Series))

            # SOME GREEK PROCESSING
            df = df.rename(columns={"strike": "STRIKE", "expiration_date": "EXPIRE_DATE"})
            df["QUOTE_READTIME"] = now.round("15min")
            df = df.convert_dtypes()
            df.drop_duplicates(
                ["QUOTE_READTIME", "STRIKE", "EXPIRE_DATE", "option_type"],
                keep="first",
                inplace=True,
            )
            df['QUOTE_DATE'] = pd.to_datetime(pd.to_datetime(df['QUOTE_READTIME']).dt.date)

            readtime = pd.to_datetime(df['QUOTE_READTIME'])

            readtime_dt = readtime.dt
            
            df['QUOTE_TIME_HOURS'] = readtime_dt.hour + readtime_dt.minute / 60

            df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])

            df['DTE'] = (df['EXPIRE_DATE'] - df['QUOTE_DATE']).dt.days + ((16 -df['QUOTE_TIME_HOURS'])/24).round(4)

            df['UNDERLYING_LAST'] = underlying_last
        except Exception as e:
            # print(f"Error: {e}")
            df = None
        return df

def format_data(df):
    """Format the DataFrame to match the sample data columns."""

    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]
    calls = calls.rename(
        columns={
            "last": "C_LAST",
            # "volume": "C_VOLUME",
            "bid": "C_BID",
            "ask": "C_ASK",
            # "strike": "STRIKE",
            # "expiration_date": "EXPIRE_DATE",
            "delta": "C_DELTA",
            "gamma": "C_GAMMA",
            "theta": "C_THETA",
            "vega": "C_VEGA",
            "rho": "C_RHO",
            # "phi": "C_PHI",
            # "QUOTE_TIME": "QUOTE_READTIME",
            "mid_iv": "C_IV",
            "open_interest": "C_OI",
        }
    ).drop(
        columns=[
            "symbol",
            "description",
            "exch",
            "type",
            "change_percentage",
            "average_volume",
            "last_volume",
            "prevclose",
            "last_volume",
            "trade_date",
            "prevclose",
            "week_52_high",
            "week_52_low",
            "bidexch",
            "bid_date",
            "askexch",
            "ask_date",
            # "open_interest",
            "contract_size",
            "expiration_type",
            "option_type",
            "root_symbol",
            "bid_iv",
            "ask_iv",
            "smv_vol",
            "updated_at",
            "change",
            "open",
            "high",
            "low",
            "close",
            "bidsize",
            "asksize",
            "underlying",
            "volume",
            "phi"
        ]
    )

    puts = puts.rename(
        columns={
            "last": "P_LAST",
            # "volume": "P_VOLUME",
            "bid": "P_BID",
            "ask": "P_ASK",
            # "strike": "STRIKE",
            # "expiration_date": "EXPIRE_DATE",
            "delta": "P_DELTA",
            "gamma": "P_GAMMA",
            "theta": "P_THETA",
            "vega": "P_VEGA",
            "rho": "P_RHO",
            # "phi": "P_PHI",
            # "QUOTE_TIME": "QUOTE_READTIME",
            "mid_iv": "P_IV",
            "open_interest": "P_OI",
        }
    ).drop(
        columns=[
            "symbol",
            "description",
            "exch",
            "type",
            "change_percentage",
            "average_volume",
            "last_volume",
            "prevclose",
            "last_volume",
            "trade_date",
            "prevclose",
            "week_52_high",
            "week_52_low",
            "bidexch",
            "bid_date",
            "askexch",
            "ask_date",
            # "open_interest",
            "contract_size",
            "expiration_type",
            "option_type",
            "root_symbol",
            "bid_iv",
            "ask_iv",
            "smv_vol",
            "updated_at",
            "change",
            "open",
            "high",
            "low",
            "close",
            "bidsize",
            "asksize",
            "underlying",
            "volume",
            "phi",
        ]
    )

    combined = pd.merge(
        calls,
        puts,
        on=["STRIKE", "EXPIRE_DATE", "QUOTE_READTIME", "UNDERLYING_LAST", "QUOTE_DATE", "QUOTE_TIME_HOURS", "DTE"],
        suffixes=("_call", "_put"),
    )
    combined.drop(columns=['QUOTE_DATE'], inplace=True)
    combined['INTEREST_RATE'] = r
    combined['C_MARK'] = (combined['C_BID'] + combined['C_ASK']) / 2
    combined['P_MARK'] = (combined['P_BID'] + combined['P_ASK']) / 2
    combined['intDTE'] = combined['DTE'].astype(int)
    combined['P_ITM'] = combined['STRIKE'].ge(combined['UNDERLYING_LAST'])
    combined['C_ITM'] = combined['STRIKE'].le(combined['UNDERLYING_LAST'])


    return combined

options = Parallel(n_jobs=16)(delayed(fetch_option_chain)(future_date) for future_date in future_dates)

options = pd.concat(options)
options['delta'] = options.apply(lambda x: delta(flag="p" if x['option_type'] == "put" else "call", S=x['UNDERLYING_LAST'], K=x['STRIKE'], t=x['DTE']/365, r=r, sigma=x['mid_iv'],q=div,), axis=1) # Matched with Option Alpha
options.sort_values(by=['QUOTE_READTIME', "DTE"], inplace=True)
options = format_data(options)
# options.columns
#### Save to parquet
year = now.year

chain_dir = "/Users/traderHuy/Downloads/SPY option backtest analysis/Tradier Option Data/schwab_chains"

if not os.path.exists(os.path.join(chain_dir, ticker)):
    os.mkdir(os.path.join(chain_dir, ticker))

if not os.path.exists(os.path.join(chain_dir, ticker, str(year))):
    os.mkdir(os.path.join(chain_dir, ticker, str(year)))

options.to_parquet(
    os.path.join(
        chain_dir,
        ticker,
        str(year),
        ticker + "_" + now.floor("15min").strftime("%Y-%m-%d %H-%M") + ".parquet",
    )
)
print(f"Saved {ticker}_{now.round('15min').strftime('%Y-%m-%d %H-%M')}.feather")
