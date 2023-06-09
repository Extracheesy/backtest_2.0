import os

import asyncio
import time

import pandas as pd

from src.crypto_data import ExchangeDataManager

import seaborn as sns
import matplotlib.pyplot as plt


"""
Stable coin:
Tether (USDT)
USD Coin (USDC)
Binance USD (BUSD)
DAI (DAI)
TrueUSD (TUSD)
Paxos Standard (PAX)
Gemini Dollar (GUSD)
HUSD (HUSD)
TerraUSD (UST)
Stably (USDS)
"""
import pycoingecko
from pycoingecko import CoinGeckoAPI

import requests

# create a CoinGeckoAPI object
cg = CoinGeckoAPI()

# use the get_coins_markets method to get the top 50 coins by market cap, excluding stablecoins
# top50_coins = cg.get_coins_markets(vs_currency='usd', per_page=50, category='coin')
top50_coins = cg.get_coins_markets(vs_currency='usd', per_page=50)

# extract the symbol for each coin and put them in a list
top50_symbols = [coin['symbol'] for coin in top50_coins]

# print the list of symbols
print(top50_symbols)

lst_stable_coin = ["USDT", "USDC", "BUSD", "DAI", "TUSD", "PAX", "GUSD", "HUSD", "UST", "USDS"]
lst_lower_stable_coin = []
for coin in lst_stable_coin:
    lst_lower_stable_coin.append(coin.lower())
lst_stable_coin = lst_lower_stable_coin

lst_symbols = []
for coin in top50_symbols:
    if not (coin in lst_stable_coin):
        lst_symbols.append(coin)

print(lst_stable_coin)
print(lst_symbols)

lst_symbols_usdt = []
for coin in lst_symbols:
    lst_symbols_usdt.append(coin.upper() + 'USDT')
print(lst_symbols_usdt)

import tracemalloc
tracemalloc.start()

# asyncio.run(exchange.download_data(["BTCUSDT"], ["1h"], start_date="2020-01-01 00:00:00"))

print(os.getcwd())
os.chdir("../")
print(os.getcwd())
exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")

lst_symbols_usdt.remove("STETHUSDT")
lst_symbols_usdt.remove("TONUSDT")
lst_symbols_usdt.remove("LEOUSDT")
lst_symbols_usdt.remove("WBTCUSDT")
lst_symbols_usdt.remove("OKBUSDT")
lst_symbols_usdt.remove("CROUSDT")
if 'FRAXUSDT' in lst_symbols_usdt:
    lst_symbols_usdt.remove("FRAXUSDT")
if 'HBTCUSDT' in lst_symbols_usdt:
    lst_symbols_usdt.remove("HBTCUSDT")

print(os.getcwd())
asyncio.run(exchange.download_data(lst_symbols_usdt, ["1h"], start_date="2020-01-01 00:00:00"))
# asyncio.run(exchange.download_data(lst_symbols_usdt, ["1h"], start_date="2021-01-01 00:00:00"))
# asyncio.run(exchange.download_data(lst_symbols_usdt, ["1h"], start_date="2022-01-01 00:00:00"))
# asyncio.run(exchange.download_data(lst_symbols_usdt, ["1h"], start_date="2023-01-01 00:00:00"))

df_symbols = pd.DataFrame()
for symbol in lst_symbols_usdt:
    filename = "./database/exchanges" + "/binance/" + "1h" + "/" + symbol + ".csv"
    df_tmp = pd.read_csv(filename)
    df_symbols[symbol] = df_tmp['open'].pct_change()

lst_corr_symbol = []
for symbol in lst_symbols_usdt:
    # lst_corr_symbol.append(symbol)
    for corr_symbol in lst_symbols_usdt:
        if corr_symbol == symbol:
            lst_corr_symbol.append(corr_symbol)
        elif df_symbols[symbol].corr(df_symbols[corr_symbol]) < 0.2:
            # print(symbol, ' no corr with ', corr_symbol, ' : ', df_symbols[symbol].corr(df_symbols[corr_symbol]))
            lst_corr_symbol.append(corr_symbol)
    print('full:    ', lst_corr_symbol[0], ' nb symbols: ', len(lst_corr_symbol), ' list: ', lst_corr_symbol)

    df_reduced = pd.DataFrame()
    for symbols_data in lst_corr_symbol:
        df_reduced[symbols_data] = df_symbols[symbols_data]

    ax = sns.heatmap(df_reduced.corr())
    plt.show()
    ax = sns.heatmap(df_reduced.corr(), cmap='RdYlGn', linewidths=.1)
    plt.show()

    """
    lst_corr_reduced = []
    for symbol_check in lst_corr_symbol:
        for corr_symbol_check in lst_corr_symbol:
            if symbol_check == corr_symbol_check:
                lst_corr_reduced.append(corr_symbol_check)
            elif df_symbols[corr_symbol_check].corr(df_symbols[corr_symbol_check]) < 0.2 \
                    and df_symbols[corr_symbol_check].corr(df_symbols[corr_symbol_check]) > -0.2:
                lst_corr_reduced.append(corr_symbol_check)
    print('reduced: ' ,lst_corr_reduced[0], ' nb symbols: ', len(lst_corr_reduced), ' list: ', lst_corr_reduced)
    """

    lst_corr_symbol = []
    lst_corr_reduced = []

ax = sns.heatmap(df_symbols.corr())
plt.show()

ax = sns.heatmap(df_symbols.corr(), cmap='RdYlGn', linewidths=.1)
plt.show()


nflx_corr_df = df_symbols.corr().NFLX
print(nflx_corr_df.idxmax())

print(nflx_corr_df[ nflx_corr_df < 1 ].idxmax())

print(nflx_corr_df.idxmin())

print('toto')






