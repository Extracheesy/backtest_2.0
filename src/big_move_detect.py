import pandas as pd
import datetime as dt
from binance.client import Client

import matplotlib.pyplot as plt

import mplfinance as mpf
import talib as ta
# import binance

import os

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

df_ohlvc = get_historical_ohlc_data("BTCUSDT", past_days=164)

df_ohlvc["date"] = pd.to_datetime(df_ohlvc["open_date_time"])
df_ohlvc.reset_index(inplace=True)
df_ohlvc.set_index('date', inplace=True)

df_ohlvc['open'] = df_ohlvc['open'].astype(float)
df_ohlvc['close'] = df_ohlvc['close'].astype(float)
df_ohlvc['high'] = df_ohlvc['high'].astype(float)
df_ohlvc['low'] = df_ohlvc['low'].astype(float)
df_ohlvc['volume'] = df_ohlvc['volume'].astype(float)

# df_ohlvc = df_ohlvc["2023-12-30":"2023-01-15"]
# df_ohlvc = df_ohlvc[:"2023-01-15"]

df_ohlvc_main = df_ohlvc.copy()


lst_idx = df_ohlvc_main.index

# df_ohlvc.to_csv("BTC_good_sample_for_big_moves.csv")

# df_ohlvc = pd.read_csv("BTC_good_sample_for_big_moves.csv")
# df_ohlvc["date"] = pd.to_datetime(df_ohlvc["date"])
# df_ohlvc.set_index('date', inplace=True)



time = "2023-05-01 04:00:00"

# open, high, low, volume, close = get_ohlcv_values(df_ohlvc, time)

# print("open: ", open, " high: ", high, " low: ", low, " volume: ", volume, "close: ", close)

# mpf.plot(df_ohlvc,type='candle',mav=(5, 3),volume=True, title='BTC')
# mpf.plot(df_ohlvc,type='candle', volume=True, title='BTC')



def identify_big_drop_rise(dataframe):
    # Calculate technical indicators
    dataframe['RSI'] = ta.momentum.RSIIndicator(dataframe['close'], window=14, fillna=True).rsi()
    dataframe['ATR'] = ta.volatility.AverageTrueRange(dataframe['high'], dataframe['low'], dataframe['close'], window=14).average_true_range()
    dataframe['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=dataframe["close"], volume=dataframe["volume"]).on_balance_volume()
    macd = ta.trend.MACD(dataframe['close'], window_slow=26, window_fast=12, window_sign=9)
    dataframe['MACD'] = macd.macd() - macd.macd_signal()
    adx = ta.trend.ADXIndicator(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
    dataframe['ADX'] = adx.adx()
    macd_histogram = ta.trend.MACD(dataframe['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
    dataframe['MACD_Histogram'] = macd_histogram
    dataframe['Stoch_RSI'] = ta.momentum.StochRSIIndicator(dataframe['RSI'], window=14).stochrsi()

    # Define line breakouts for big drop/rise
    rsi_breakout = 50  # RSI breakout value
    stoch_rsi_breakout = 0.8  # Stochastic RSI breakout value
    atr_breakout = ta.trend.sma_indicator(dataframe['ATR'], window=14)  # ATR breakout as 14-day SMA
    obv_breakout = ta.trend.ema_indicator(dataframe['OBV'], window=20)  # OBV breakout as 20-day EMA
    macd_breakout = ta.trend.sma_indicator(dataframe['MACD'], window=9)  # MACD breakout as 9-day SMA
    adx_breakout = 30  # ADX breakout value
    macd_histogram_breakout = 0  # MACD Histogram breakout value

    # Calculate trend or slope of indicators
    dataframe['OBV_Trend'] = ta.trend.sma_indicator(dataframe['OBV'], window=5).diff() > 0  # OBV trend as positive slope of 5-day SMA
    dataframe['RSI_Trend'] = ta.trend.sma_indicator(dataframe['RSI'], window=5).diff() > 0  # RSI trend as positive slope of 5-day SMA
    dataframe['Stoch_RSI_Trend'] = ta.trend.sma_indicator(dataframe['Stoch_RSI'], window=5).diff() > 0  # Stochastic RSI trend as positive slope of 5-day SMA
    dataframe['ADX_Trend'] = ta.trend.sma_indicator(dataframe['ADX'], window=5).diff() > 0  # ADX trend as positive slope of 5-day SMA
    dataframe['MACD_Histogram_Trend'] = ta.trend.sma_indicator(dataframe['MACD_Histogram'], window=5).diff() > 0  # MACD Histogram trend as positive slope of 5-day SMA

    # Check for big drop or big rise
    dataframe['Big_Drop'] = (dataframe['RSI'].shift(1) > rsi_breakout) & (dataframe['RSI'] <= rsi_breakout)
    dataframe['Big_Rise'] = (dataframe['RSI'].shift(1) < (100 - rsi_breakout)) & (dataframe['RSI'] >= (100 - rsi_breakout))
    dataframe['Big_Drop'] |= (dataframe['Stoch_RSI'].shift(1) > stoch_rsi_breakout) & (dataframe['Stoch_RSI'] <= stoch_rsi_breakout)
    dataframe['Big_Rise'] |= (dataframe['Stoch_RSI'].shift(1) < (1 - stoch_rsi_breakout)) & (dataframe['Stoch_RSI'] >= (1 - stoch_rsi_breakout))
    dataframe['Big_Drop'] |= (dataframe['ATR'].shift(1) > atr_breakout) & (dataframe['ATR'] <= atr_breakout)
    dataframe['Big_Rise'] |= (dataframe['ATR'].shift(1) < atr_breakout) & (dataframe['ATR'] >= atr_breakout)
    dataframe['Big_Drop'] |= (dataframe['OBV'].shift(1) > obv_breakout) & (dataframe['OBV'] <= obv_breakout)
    dataframe['Big_Rise'] |= (dataframe['OBV'].shift(1) < obv_breakout) & (dataframe['OBV'] >= obv_breakout)
    dataframe['Big_Drop'] |= (dataframe['MACD'].shift(1) > macd_breakout) & (dataframe['MACD'] <= macd_breakout)
    dataframe['Big_Rise'] |= (dataframe['MACD'].shift(1) < macd_breakout) & (dataframe['MACD'] >= macd_breakout)
    dataframe['Big_Drop'] |= (dataframe['ADX'].shift(1) > adx_breakout) & (dataframe['ADX'] <= adx_breakout)
    dataframe['Big_Rise'] |= (dataframe['ADX'].shift(1) < adx_breakout) & (dataframe['ADX'] >= adx_breakout)
    dataframe['Big_Drop'] |= (dataframe['MACD_Histogram'].shift(1) > macd_histogram_breakout) & (dataframe['MACD_Histogram'] <= macd_histogram_breakout)
    dataframe['Big_Rise'] |= (dataframe['MACD_Histogram'].shift(1) < macd_histogram_breakout) & (dataframe['MACD_Histogram'] >= macd_histogram_breakout)

    sma_x = 1.5
    sma_window = 24

    dataframe['pct_change_close'] = dataframe['close'].pct_change().apply(abs)
    dataframe['sma_change_close'] = ta.trend.sma_indicator(dataframe['pct_change_close'], window=sma_window) * sma_x
    dataframe['delta'] = dataframe['high'] - dataframe['low']
    dataframe['sma_delta'] = ta.trend.sma_indicator(dataframe['delta'], window=sma_window) * sma_x
    dataframe['pct_change_delta'] = dataframe['delta'].pct_change().apply(abs)
    dataframe['sma_pct_delta'] = ta.trend.sma_indicator(dataframe['pct_change_delta'], window=sma_window) * sma_x
    dataframe['pct_change_volume'] = dataframe['volume'].pct_change().apply(abs)
    dataframe['sma_pct_volume'] = ta.trend.sma_indicator(dataframe['pct_change_volume'], window=sma_window) * sma_x
    dataframe['volume'] = dataframe['volume'] / 1000
    dataframe['sma_volume'] = ta.trend.sma_indicator(dataframe['volume'], window=sma_window) * sma_x

    dataframe['my_big_move_detector'] = (dataframe['pct_change_close'] > dataframe['sma_change_close']) \
                                        & (dataframe['pct_change_volume'] > dataframe['sma_pct_volume']) \
                                        & (dataframe['volume'] > dataframe['sma_volume'])
                                        # & (dataframe['sma_delta'] > dataframe['pct_change_delta']) \
                                        # & (dataframe['pct_change_delta'] > dataframe['sma_pct_delta']) \
    duration = 3    # 3 hours

    for i in range(0, duration):
        dataframe['my_big_move_detector' + "_n" + str(i)] = dataframe['my_big_move_detector'].shift(i)

    for i in range(0, duration):
        dataframe['my_big_move_detector'] |= dataframe['my_big_move_detector_n' + str(i)]

    dataframe.dropna(inplace=True)

    return dataframe


directory = "test_big_move/"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")

# Delete all files in the directory if it exists
if os.path.exists(directory):
    file_list = os.listdir(directory)
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)
    print(f"All files in directory '{directory}' deleted.")
else:
    print(f"Directory '{directory}' doesn't exist.")

idx = 0
len_week = 168
week_id = 1
while idx < len(lst_idx):
    start_date = lst_idx[idx]
    try:
        end_date = lst_idx[min(idx+len_week + 24, len(lst_idx))]
    except:
        end_date = lst_idx[min(idx+len_week + 24, len(lst_idx) - 1)]
    df_ohlvc = df_ohlvc_main[start_date : end_date].copy()
    idx = idx + len_week

    fig, axs = plt.subplots(figsize=(15, 20), nrows=8, ncols=1)  # Create a (4, 1) subplot grid

    # Create the mplfinance plot in the first subplot
    mpf.plot(df_ohlvc, ax=axs[0], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed
    # mpf.plot(df_ohlvc, ax=axs[1], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed
    # mpf.plot(df_ohlvc, ax=axs[2], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed
    # mpf.plot(df_ohlvc, ax=axs[3], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

    df_ohlvc = identify_big_drop_rise(df_ohlvc)

    axs[1].plot(df_ohlvc['RSI'], lw=1)
    axs[2].plot(df_ohlvc['ATR'], lw=1)
    axs[3].plot(df_ohlvc['OBV'], lw=1)
    axs[4].plot(df_ohlvc['MACD'], lw=1)
    axs[5].plot(df_ohlvc['ADX'], lw=1)
    axs[6].plot(df_ohlvc['MACD_Histogram'], lw=1)
    axs[7].plot(df_ohlvc['Stoch_RSI'], lw=1)

    plt.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots if necessary

    plt.suptitle("Subplots Example")

    # plt.show()  # Display the combined plot with all subplots
    plt.savefig(directory + "plot_1_week_" + str(week_id) + ".png")


    """
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(df_ohlvc.index, df_ohlvc['Big_Drop'].astype(int), width=0.8, color='red', alpha=0.5, label='Big Drop')
    ax.bar(df_ohlvc.index, df_ohlvc['Big_Rise'].astype(int), width=0.8, color='green', alpha=0.5, label='Big Rise')
    ax.set_xlabel('Date')
    ax.set_ylabel('Signal')
    ax.set_title('Big Drop and Big Rise Signals')
    ax.legend()
    plt.show()
    """

    fig, axs = plt.subplots(figsize=(15, 20), nrows=8, ncols=1)  # Create a (4, 1) subplot grid

    # Create the mplfinance plot in the first subplot
    mpf.plot(df_ohlvc, ax=axs[0], type='candle', volume=False , axtitle='BTC' )  # Modify plot parameters as needed

    df_ohlvc = df_ohlvc.reset_index()

    axs[1].bar(df_ohlvc.index,df_ohlvc['my_big_move_detector'], color='blue', label='close')

    axs[2].bar(df_ohlvc.index,df_ohlvc['pct_change_close'], color='blue', label='close')
    axs[2].plot(df_ohlvc.index,df_ohlvc['sma_change_close'], color='orange', label='sma_close')
    axs[2].set_title('close')
    axs[3].bar(df_ohlvc.index,df_ohlvc['pct_change_delta'], color='yellow', label='delta')
    axs[3].plot(df_ohlvc.index,df_ohlvc['sma_pct_delta'], color='orange', label='sma_close')
    axs[3].set_title('pct delta')
    axs[4].bar(df_ohlvc.index,df_ohlvc['pct_change_volume'], color='purple', label='volume')
    axs[4].plot(df_ohlvc.index,df_ohlvc['sma_pct_volume'], color='orange', label='sma_close')
    axs[4].set_title('pct volume')
    axs[5].bar(df_ohlvc.index,df_ohlvc['volume'], color='brown', label='volume')
    axs[5].plot(df_ohlvc.index,df_ohlvc['sma_volume'], color='orange', label='sma_close')
    axs[5].set_title('volume')

    axs[6].bar(df_ohlvc.index,df_ohlvc['Big_Drop'].astype(int), color='red', label='Big Drop')
    axs[6].set_title('Big Drop')
    axs[7].bar(df_ohlvc.index,df_ohlvc['Big_Rise'].astype(int), color='green', label='Big Rise')
    axs[7].set_title('Big Rise')

    # df_ohlvc['Big_Rise_int'] = df_ohlvc['Big_Rise'].replace({True: 1, False: 0})
    # df_ohlvc['Big_Drop_int'] = df_ohlvc['Big_Drop'].replace({True: 1, False: 0})

    # axs[1].bar(df_ohlvc['Big_Rise_int'], lw=1)
    # axs[2].bar(df_ohlvc['Big_Drop_int'], lw=1)

    plt.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots if necessary

    plt.suptitle("Subplots Example")

    # plt.show()  # Display the combined plot with all subplot
    plt.savefig(directory + "plot_2_week_" + str(week_id) + ".png")

    week_id += 1















