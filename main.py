# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from utilities.get_data import get_historical_from_db
from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from src.bol_trend import BolTrend
from src.mean_reversion import MeanReversion
from src.mean_bol_trend import MeanBolTrend
from src.rsi_bb_sma import RSI_BB_SMA
from src.bol_trend_live import BolTrendLive
from src.hull_suite import HullSuite
from src.cluc_may import ClucMay
from src.obv import Obv
from src.bigwill import BigWill
from src.turtle import Turtle
from src.scalping_engulfing import ScalpingEngulfing

from src.cryptobot_indicators import TechnicalAnalysis

from src.slope_is_dope import SlopeIsDope
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

    lst_symbol = ['BTC', 'ETH', 'XRP', 'EOS', 'BCH', 'LTC', 'ADA', 'ETC', 'LINK', 'TRX', 'DOT', 'DOGE', 'SOL', 'MATIC', 'BNB', 'UNI',
     'ICP', 'AAVE', 'FIL', 'XLM', 'ATOM', 'XTZ', 'SUSHI', 'AXS', 'THETA', 'AVAX', 'DASH', 'SHIB', 'MANA', 'GALA',
     'SAND', 'DYDX', 'CRV', 'NEAR', 'EGLD', 'KSM', 'AR', 'REN', 'FTM', 'PEOPLE', 'LRC', 'NEO', 'ALICE', 'WAVES', 'ALGO',
     'IOTA', 'YFI', 'ENJ', 'GMT', 'ZIL', 'IOST', 'APE', 'RUNE', 'KNC', 'APT', 'CHZ', 'XMR', 'ROSE', 'ZRX', 'KAVA',
     'ENS', 'GAL', 'MTL', 'AUDIO', 'SXP', 'C98', 'OP', 'RSR', 'SNX', 'STORJ', '1INCH', 'COMP', 'IMX', 'LUNA2', 'FLOW',
     'REEF', 'TRB', 'QTUM', 'API3', 'MASK', 'WOO', 'GRT', 'BAND', 'STG', 'LUNC', 'ONE', 'JASMY', 'FOOTBALL', 'MKR',
     'BAT', 'MAGIC', 'ALPHA', 'LDO', 'OCEAN', 'CELO', 'BLUR', 'MINA', 'CORE', 'CFX', 'HIGH', 'ASTR', 'AGIX', 'GMX',
     'LINA', 'ANKR', 'GFT', 'ACH', 'FET', 'FXS', 'RNDR', 'HOOK', 'BNX', 'SSV', 'BGHOT10', 'USDC', 'LQTY', 'STX', 'TRU',
     'DUSK', 'HBAR', 'INJ', 'BEL', 'COTI', 'VET', 'ARB', 'TOMO', 'LOOKS', 'KLAY', 'FLM', 'OMG', 'RLC', 'CKB', 'ID',
     'LIT', 'JOE', 'TLM', 'HOT', 'BLZ', 'CHR', 'RDNT', 'ICX', 'HFT', 'ONT', 'ZEC', 'UNFI', 'NKN', 'ARPA', 'DAR', 'SFP',
     'CTSI', 'SKL', 'RVN', 'CELR', 'FLOKI', 'SPELL', 'SUI', 'EDU', 'PEPE', 'METAHOT', 'IOTX', 'CTK', 'STMX', 'UMA',
     'BSV', '10000AIDOGE', '10000LADYS', 'TON', 'GTC', 'DENT', 'ZEN', 'PHB', 'ORDI', 'KEY', 'IDEX', 'SLP']

    # pair = "BTC/USDT"
    pair = "ETH/USDT"
    tf = "1h"
    # tf = "5m"
    # tf = "1m"

    print(os.getcwd())

    if True:
        exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")
        # asyncio.run(exchange.download_data(["BTCUSDT"], [tf], start_date="2020-01-01 00:00:00"))
        # asyncio.run(exchange.download_data(["ETHUSDT"], ["5m"], start_date="2020-01-01 00:00:00"))
        asyncio.run(exchange.download_data(["ETHUSDT"], [tf], start_date="2020-01-01 00:00:00"))


        # asyncio.run(exchange.download_data(["BTCUSDT", "ETHUSDT"], ["1m", "1h"], start_date="2020-01-01 00:00:00"))
        # eth_histo = exchange.load_data('ETHUSDT', '1h', start_date="2020-01-01")
        # btc_histo = exchange.load_data('BTCUSDT', '1m', start_date="2020-01-01")

    df = get_historical_from_db(
        ccxt.binance(),
        pair,
        tf,
        path="./database/"
    )

    model = TechnicalAnalysis(df)
    model.add_all()
    model.add_candles()
    print(model.df.columns.tolist())

    # STRATEGY = "SCALPING"

    #     STRATEGY = "CLUCKMAY"
    STRATEGY = "BOLTREND"
    STRATEGY = "OBV"
    STRATEGY = "TURTLE"

    STRATEGY = "BIGWILL"



    # STRATEGY = "MEANBOLTREND"
    # STRATEGY = "MEANREVERSION"
    # STRATEGY = "RSI_BB_SMA"
    # STRATEGY = "BOLTRENDLIVE"


    # STRATEGY = "HULLSUITE"

    if STRATEGY == "SCALPING":
        strat = ScalpingEngulfing(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # df=df.iloc[-10000:],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],
            bol_window=100,
            bol_std=2.25,
            min_bol_spread=0,
            long_ma_window=200,
        )
    elif STRATEGY == "BOLTREND":
        strat = BolTrend(
            # df=df.loc["2018":],
            df=df,
            # df=df.loc["2022":],
            # df=df.loc["2023":],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],

            # bol_window=100,
            bol_window=20,
            bol_std=2.25,
            # bol_std=2.0,

            min_bol_spread=0,
            long_ma_window=100,

            SL = 0,
            TP = 0
        )
    elif STRATEGY == "MEANBOLTREND":
        strat = MeanBolTrend(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2022":],
            df=df.loc["2023":],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],

            # bol_window=100,
            bol_window=20,
            # bol_std=2.25,
            bol_std=2.0,

            min_bol_spread=0,
            long_ma_window=100,

            SL=-10,
            TP=10
        )
    elif STRATEGY == "MEANREVERSION":
        strat = MeanReversion(
            # df=df.loc["2018":],
            # df=df,
            df=df.loc["2022":],
            # df=df.loc["2023":],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],

            # bol_window=100,
            bol_window=20,
            # bol_std=2.25,
            bol_std=2.0,

            min_bol_spread=0,
            long_ma_window=100,

            SL=-5,
            TP=0
        )
    elif STRATEGY == "RSI_BB_SMA":
        strat = RSI_BB_SMA(
            # df=df.loc["2018":],
            df=df,
            # df=df.loc["2022":],
            # df=df.loc["2023":],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],
            bol_window=100,
            bol_std=2.25,
            min_bol_spread=0,
            long_ma_window=500,
        )
    elif STRATEGY == "BOLTRENDLIVE":
        strat = BolTrendLive(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2022":],
            df=df.loc["2023":],
            type=["long", "short"],
            # type=["long"],
            # type=["short"],
            bol_window=100,
            bol_std=2.25,
            min_bol_spread=0,
            long_ma_window=500,
        )
    elif STRATEGY == "CLUCKMAY":
        strat = ClucMay(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # type=["short"],
            type=["long"],
            # type=["long", "short"],
            bol_window=20,
            bol_std=3,
            min_bol_spread=0,
            long_ma_window=100,
            rsi_timeperiod=2,
            ema_rsi_timeperiod=5,
            ema_timeperiod=100,

            SL=0,
            TP=0
        )
    elif STRATEGY == "HULLSUITE":
        strat = HullSuite(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # type=["short"],
            # type=["long"],
            type=["long", "short"],
            short_window = 9,
            long_window = 18,

            SL=0,
            TP=0
        )
    elif STRATEGY == "OBV":
        strat = Obv(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # type=["short"],
            # type=["long"],
            type=["long", "short"],
            ema_window = 200,

            SL=0,
            TP=0
        )
    elif STRATEGY == "TURTLE":
        strat = Turtle(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # type=["short"],
            # type=["long"],
            type=["long", "short"],
            SL=0,
            TP=0
        )
    elif STRATEGY == "BIGWILL":
        strat = BigWill(
            # df=df.loc["2018":],
            # df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            df=df.loc["2023":],
            # type=["short"],
            type=["long"],
            # type=["long", "short"],
            SL=0,
            TP=0
        )

    """
    strat = SlopeIsDope(
        df=df.loc["2018":],
        type=["long"],
    )
    """

    strat.populate_indicators()
    strat.populate_buy_sell()
    # strat.populate_sltp()
    bt_result = strat.run_backtest(initial_wallet=1000, leverage=5)
    df_trades, df_days = basic_single_asset_backtest(trades=bt_result['trades'], days=bt_result['days'])

    df_trades.to_csv("trades.csv")

    plot_wallet_vs_asset(df_days=df_days)

    df_trades

    plot_bar_by_month(df_days=df_days)


    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

