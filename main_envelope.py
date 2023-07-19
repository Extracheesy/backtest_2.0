# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import conf.config
from utilities.get_data import get_historical_from_db
from utilities.backtesting import basic_single_asset_backtest_with_df, basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from src.bol_trend import BolTrend
from src.mean_reversion import MeanReversion
from src.mean_bol_trend import MeanBolTrend
from src.rsi_bb_sma import RSI_BB_SMA
from src.bol_trend_live import BolTrendLive
from src.hull_suite import HullSuite
from src.envelope import Envelope
from src.cluc_may import ClucMay
from src.scalping_engulfing import ScalpingEngulfing
from src.analyse_pair import AnalysePair
from src.cryptobot_indicators import TechnicalAnalysis

from src.slope_is_dope import SlopeIsDope
import ccxt
import numpy as np

import os
import glob
import asyncio

from src.crypto_data import ExchangeDataManager

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lst_symbol = [
        'BTC', 'ETH', 'XRP', 'EOS', 'BCH', 'LTC', 'ADA', 'ETC', 'LINK', 'TRX', 'DOT', 'DOGE', 'SOL', 'MATIC', 'BNB', 'UNI',
        'ICP', 'AAVE', 'FIL', 'XLM', 'ATOM',
        'XTZ', 'SUSHI', 'AXS', 'THETA', 'AVAX', 'DASH', 'SHIB', 'MANA', 'GALA', 'SAND', 'DYDX', 'CRV', 'NEAR', 'EGLD', 'KSM',
        'AR', 'REN', 'FTM', 'PEOPLE', 'LRC', 'NEO', 'ALICE', 'WAVES', 'ALGO',
     'IOTA', 'YFI', 'ENJ', 'GMT', 'ZIL', 'IOST', 'APE', 'RUNE', 'KNC', 'APT', 'CHZ', 'XMR', 'ROSE', 'ZRX', 'KAVA',
     'ENS', 'GAL', 'MTL', 'AUDIO', 'SXP', 'C98', 'OP', 'RSR', 'SNX', 'STORJ', '1INCH', 'COMP', 'IMX', 'LUNA2', 'FLOW',
     'REEF', 'TRB', 'QTUM', 'API3', 'MASK', 'WOO', 'GRT', 'BAND', 'STG', 'LUNC', 'ONE', 'JASMY', 'FOOTBALL', 'MKR',
     'BAT', 'MAGIC', 'ALPHA', 'LDO', 'OCEAN', 'CELO', 'BLUR', 'MINA',
                  # 'CORE',
                  'CFX', 'HIGH', 'ASTR', 'AGIX', 'GMX',
     'LINA', 'ANKR',
                  # 'GFT',
                  'ACH', 'FET', 'FXS', 'RNDR', 'HOOK', 'BNX', 'SSV',
                  # 'BGHOT10',
                  'LQTY', 'STX', 'TRU',
     'DUSK', 'HBAR', 'INJ', 'BEL', 'COTI', 'VET', 'ARB', 'TOMO',
                  # 'LOOKS',
                  'KLAY', 'FLM', 'OMG', 'RLC', 'CKB', 'ID',
     'LIT', 'JOE', 'TLM', 'HOT', 'BLZ', 'CHR', 'RDNT', 'ICX', 'HFT', 'ONT', 'ZEC', 'UNFI', 'NKN', 'ARPA', 'DAR', 'SFP',
     'CTSI', 'SKL', 'RVN', 'CELR', 'FLOKI', 'SPELL', 'SUI', 'EDU', 'PEPE',
                  # 'METAHOT',
                  'IOTX', 'CTK', 'STMX', 'UMA',
     # 'BSV',
                  # '10000AIDOGE',
                  # '10000LADYS',
                  # 'TON',
                  'GTC', 'DENT', 'ZEN', 'PHB',
                  # 'ORDI',
                  'KEY', 'IDEX', 'SLP']

    # lst_symbol = ['BTC','ETH']
    # lst_symbol = ['XTZ', 'SUSHI']

    lst_pair = []
    for symbol in lst_symbol:
        lst_pair.append(symbol + "/USDT")

    tf = "1h"
    start = "2023-01-01 00:00:00"
    if False:
        try:
            exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")
            asyncio.run(exchange.download_data(lst_pair, [tf], start_date=start))
        except:
            print("download failure")

    for offset in [2,3,4,5,6,7]:
        analyser = AnalysePair(
            envelope_window = 5,
            envelope_offset = offset
        )
        for pair in lst_pair:
            df = get_historical_from_db(
                ccxt.binance(),
                pair,
                tf,
                path="./database/"
            )

            analyser.volatility_analyse_envelope_crossing(df, pair)

        analyser.store_results()

    list_files = glob.glob('envelope_*_analyse_volatility_results.csv')

    df_results = pd.DataFrame()
    for file in list_files:
        df_tmp = pd.read_csv(file)
        if len(df_results) == 0:
            df_results = df_tmp.copy()
        else:
            df_results = pd.concat([df_results, df_tmp], ignore_index=True, sort=False)
        os.remove(file)

    try:
        df_results.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        pass
    try:
        df_results.to_csv("envelope_final_results.csv")
    except:
        df_results.to_csv("envelope_final_results_2.csv")

    df_final_results = pd.DataFrame()
    for offset in [2,3,4,5]:
    # for offset in [3]:
        for pair in lst_pair:
            df = get_historical_from_db(
                ccxt.binance(),
                pair,
                tf,
                path="./database/"
            )

            strat = Envelope(
                # df=df.loc["2018":],
                # df=df,
                # df=df.loc["2021":],
                # df=df.loc["2022":],
                df=df.loc["2023":],
                # type=["short"],
                # type=["long"],
                type=["long", "short"],
                envelope_offset=offset,
                envelope_window=5,
                SL=0,
                TP=0
            )

            strat.populate_indicators()
            strat.populate_buy_sell()
            # strat.populate_sltp()
            bt_result = strat.run_backtest(initial_wallet=1000, leverage=5)
            if conf.config.PRINT_OUT:
                print("pair: ", pair, " offset: ", offset)
            if bt_result != None:
                df_trades, df_days , df_tmp = basic_single_asset_backtest_with_df(trades=bt_result['trades'], days=bt_result['days'])
            else:
                lst_columns = ["initial_wallet", "final_wallet", "vs_usd_pct", "sharpe_ratio", "max_trades_drawdown", "max_days_drawdown", "buy_and_hold_pct", "vs_hold_pct", "total_trades", "global_win_rate", "avg_profit", "total_fees"]
                lst_data_nul = [1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                df_tmp = pd.DataFrame(columns=lst_columns)
                df_tmp.loc[len(df_tmp)] = lst_data_nul

            df_tmp['pair'] = pair
            df_tmp['offset'] = offset

            if len(df_final_results) == 0:
                df_final_results = df_tmp.copy()
            else:
                df_final_results = pd.concat([df_final_results, df_tmp], ignore_index=True, sort=False)
    try:
        df_final_results.to_csv("envelope_final_global_results.csv")
    except:
        df_final_results.to_csv("envelope_final_global_results_2.csv")

    while(True):
        print("toto")
        exit()

    print(os.getcwd())
    for symbol in lst_symbol:
        pair = symbol + "/USDT"
        tf = "1h"
        # tf = "5m"
        # tf = "1m"
        try:
            exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")
            # asyncio.run(exchange.download_data(["BTCUSDT"], [tf], start_date="2020-01-01 00:00:00"))
            # asyncio.run(exchange.download_data(["ETHUSDT"], ["5m"], start_date="2020-01-01 00:00:00"))
            # asyncio.run(exchange.download_data(["ETHUSDT"], [tf], start_date="2020-01-01 00:00:00"))
            asyncio.run(exchange.download_data([pair], [tf], start_date="2023-01-01 00:00:00"))

            # asyncio.run(exchange.download_data(["BTCUSDT", "ETHUSDT"], ["1m", "1h"], start_date="2020-01-01 00:00:00"))
            # eth_histo = exchange.load_data('ETHUSDT', '1h', start_date="2020-01-01")
            # btc_histo = exchange.load_data('BTCUSDT', '1m', start_date="2020-01-01")

            df = get_historical_from_db(
                ccxt.binance(),
                pair,
                tf,
                path="./database/"
            )
        except:
            print("symbol failure: ", pair)






    model = TechnicalAnalysis(df)
    model.add_all()
    model.add_candles()
    print(model.df.columns.tolist())

    # STRATEGY = "SCALPING"

    #     STRATEGY = "CLUCKMAY"
    # STRATEGY = "BOLTREND"

    # STRATEGY = "MEANBOLTREND"
    # STRATEGY = "MEANREVERSION"
    # STRATEGY = "RSI_BB_SMA"
    # STRATEGY = "BOLTRENDLIVE"


    # STRATEGY = "HULLSUITE"
    STRATEGY = "ENVELOPE"

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
            # df=df,
            # df=df.loc["2022":],
            df=df.loc["2023":],
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
    elif STRATEGY == "ENVELOPE":
        strat = Envelope(
            # df=df.loc["2018":],
            df=df,
            # df=df.loc["2021":],
            # df=df.loc["2022":],
            # df=df.loc["2023":],
            # type=["short"],
            # type=["long"],
            type=["long", "short"],
            envelope_offset=3,
            envelope_window=5,
            SL=0,
            TP=0
        )



    strat.populate_indicators()
    strat.populate_buy_sell()
    # strat.populate_sltp()
    bt_result = strat.run_backtest(initial_wallet=1000, leverage=5)
    df_trades, df_days = basic_single_asset_backtest(trades=bt_result['trades'], days=bt_result['days'])

    # df_trades.to_csv("trades.csv")

    plot_wallet_vs_asset(df_days=df_days)

    # df_trades

    plot_bar_by_month(df_days=df_days)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

