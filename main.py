# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from utilities.get_data import get_historical_from_db
from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from src.bol_trend import BolTrend
import ccxt
import numpy as np

import os

import asyncio

from src.crypto_data import ExchangeDataManager

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pair = "BTC/USDT"
    tf = "1h"

    print(os.getcwd())

    if False:
        exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")
        asyncio.run(exchange.download_data(["BTCUSDT"], ["1h"], start_date="2020-01-01 00:00:00"))

        # asyncio.run(exchange.download_data(["BTCUSDT", "ETHUSDT"], ["1m", "1h"], start_date="2020-01-01 00:00:00"))
        # eth_histo = exchange.load_data('ETHUSDT', '1h', start_date="2020-01-01")
        # btc_histo = exchange.load_data('BTCUSDT', '1m', start_date="2020-01-01")

    df = get_historical_from_db(
        ccxt.binance(),
        pair,
        tf,
        path="./database/"
    )

    strat = BolTrend(
        df=df.loc["2018":],
        type=["long", "short"],
        bol_window=100,
        bol_std=2.25,
        min_bol_spread=0,
        long_ma_window=500,
    )

    strat.populate_indicators()
    strat.populate_buy_sell()
    bt_result = strat.run_backtest(initial_wallet=1000, leverage=1)
    df_trades, df_days = basic_single_asset_backtest(trades=bt_result['trades'], days=bt_result['days'])
    plot_wallet_vs_asset(df_days=df_days)

    df_trades

    plot_bar_by_month(df_days=df_days)


    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
