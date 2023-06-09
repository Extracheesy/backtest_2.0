import pandas as pd
import datetime as dt
from binance.client import Client

import matplotlib.pyplot as plt

import mplfinance as mpf

# import binance

client = Client()

def get_historical_ohlc_data(symbol, past_days=None, interval=None):
    """Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data"""

    if not interval:
        interval = '1h'  # default interval 1 hour
    if not past_days:
        past_days = 30  # default past days 30.

    start_str = str((pd.to_datetime('today') - pd.Timedelta(str(past_days) + ' days')).date())

    client.get_historical_klines(symbol=symbol, start_str=start_str, interval=interval)
    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, interval=interval))
    D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                 'taker_base_vol', 'taker_quote_vol', 'is_best_match']
    D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
    D['symbol'] = symbol
    D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
           'taker_quote_vol']]

    return D


def get_ohlcv_values(df_ohlvc, time):
    open = df_ohlvc.at[time, 'open']
    close = df_ohlvc.at[time, 'close']
    high = df_ohlvc.at[time, 'high']
    low = df_ohlvc.at[time, 'low']
    volume = df_ohlvc.at[time, 'volume']

    return open, high, low, volume, close

df_ohlvc = get_historical_ohlc_data("BTCUSDT", past_days=2)

print(df_ohlvc)

df_ohlvc["date"] = pd.to_datetime(df_ohlvc["open_date_time"])
df_ohlvc.reset_index(inplace=True)
df_ohlvc.set_index('date', inplace=True)

df_ohlvc['open'] = df_ohlvc['open'].astype(float)
df_ohlvc['close'] = df_ohlvc['close'].astype(float)
df_ohlvc['high'] = df_ohlvc['high'].astype(float)
df_ohlvc['low'] = df_ohlvc['low'].astype(float)
df_ohlvc['volume'] = df_ohlvc['volume'].astype(float)

time = "2023-05-01 04:00:00"

# open, high, low, volume, close = get_ohlcv_values(df_ohlvc, time)

# print("open: ", open, " high: ", high, " low: ", low, " volume: ", volume, "close: ", close)

# mpf.plot(df_ohlvc,type='candle',mav=(5, 3),volume=True, title='BTC')
mpf.plot(df_ohlvc,type='candle', volume=True, title='BTC')





fig, axs = plt.subplots(figsize=(15, 20), nrows=4, ncols=1)  # Create a (4, 1) subplot grid

# Create the mplfinance plot in the first subplot
mpf.plot(df_ohlvc, ax=axs[0], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

mpf.plot(df_ohlvc, ax=axs[1], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

mpf.plot(df_ohlvc, ax=axs[2], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

mpf.plot(df_ohlvc, ax=axs[3], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

# Customize the other subplots
# axs[1].plot(x_data1, y_data1)  # Example line plot on the second subplot
# axs[2].bar(x_data2, y_data2)   # Example bar plot on the third subplot
# axs[3].scatter(x_data3, y_data3)  # Example scatter plot on the fourth subplot

plt.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots if necessary

plt.suptitle("Subplots Example")

plt.show()  # Display the combined plot with all subplots
