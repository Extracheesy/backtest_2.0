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
from src.bigwill import BigWill
from src.bollinger_reversion import BollingerReversion
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

import matplotlib.pyplot as plt

from src.crypto_data import ExchangeDataManager

def replace_in_list(original_list, string_to_replace, replacement_string):
    modified_list = []
    for item in original_list:
        modified_item = item.replace(string_to_replace, replacement_string)
        modified_list.append(modified_item)
    return modified_list


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if False:
        dt_tmp_test = pd.read_csv('nb_engaged.csv')

        dt_tmp_test['X'] = dt_tmp_test.index

        # Plot the 'Count' column as a bar chart
        dt_tmp_test.plot(x='X', y='nb_engaged', kind='bar')

        # Set labels and title
        plt.xlabel('nb symbol engaged with bigwill')
        plt.ylabel('Count')
        plt.title('Big Will')

        # Display the plot
        plt.show()

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
                  'LQTY', 'STX',
        # 'TRU',
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

    for offset in [2, 3, 4, 5, 6]:
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
        df_results.to_csv("bollinger_reversion_final_results.csv")
    except:
        df_results.to_csv("bollinger_reversion_final_results_2.csv")

    df_global_engaged = pd.DataFrame()

    # lst_stop_loss = [0, -2, -5, -7, -10]
    lst_stop_loss = [0, -10]
    # lst_offset = [2, 3, 4, 5, 6]
    lst_offset = [2]
    lst_stochOverBought = [0.7, 0.8, 0.9, 0.95]
    lst_stochOverSold = [0.05, 0.1, 0.2, 0.3]
    lst_willOverSold = [-70, -80, -90, -95]
    lst_willOverBought = [-30, -20, -10, -5]

    df_final_results = pd.DataFrame()
    for sl in lst_stop_loss:
        for stochOverBought in lst_stochOverBought:
            for stochOverSold in lst_stochOverSold:
                for offset in lst_offset:
                    for willOverSold in lst_willOverSold:
                        for willOverBought in lst_willOverBought:
                            for pair in lst_pair:
                                df = get_historical_from_db(
                                    ccxt.binance(),
                                    pair,
                                    tf,
                                    path="./database/"
                                )

                                strat = BollingerReversion(
                                    # df=df.loc["2018":],
                                    # df=df,
                                    # df=df.loc["2021":],
                                    # df=df.loc["2022":],
                                    df=df.loc["2023":],
                                    # type=["short"],
                                    # type=["long"],
                                    type=["long", "short"],
                                    bol_window=100,
                                    bol_std=2.25,
                                    min_bol_spread=0,
                                    long_ma_window=500,
                                    stochOverBought=stochOverBought,
                                    stochOverSold=stochOverSold,
                                    willOverSold=willOverSold,
                                    willOverBought=willOverBought,

                                    SL=sl,
                                    TP=0
                                )

                                strat.populate_indicators()
                                strat.populate_buy_sell()
                                # strat.populate_sltp()
                                bt_result = strat.run_backtest(initial_wallet=1000, leverage=5)

                                ENGAGED = False
                                if ENGAGED:
                                    if bt_result != None:
                                        df_engaged = strat.get_df_engaged(pair, bt_result)

                                        if len(df_global_engaged) == 0:
                                            df_global_engaged = df_engaged
                                        else:
                                            serie_tmp = df_engaged[pair].copy()
                                            df_global_engaged[pair] = df_engaged[pair]

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
                                df_tmp['stop_loss'] = sl
                                df_tmp["stochOverBought"] = stochOverBought
                                df_tmp["stochOverSold"] = stochOverSold
                                df_tmp["offset"] = offset
                                df_tmp["willOverSold"] = willOverSold
                                df_tmp["willOverBought"] = willOverBought

                                if len(df_final_results) == 0:
                                    df_final_results = df_tmp.copy()
                                else:
                                    df_final_results = pd.concat([df_final_results, df_tmp], ignore_index=True, sort=False)

    df_final_results.drop(df_final_results[df_final_results['final_wallet'] <= 1500].index, inplace = True)
    df_final_results.drop(df_final_results[df_final_results['vs_hold_pct'] <= 0.1].index, inplace = True)
    df_final_results.drop(df_final_results[df_final_results['global_win_rate'] <= 0.6].index, inplace = True)

    df_final_results.to_csv("bollinger_reversion_final_global_results.csv")

    lst_columns = ['stop_loss', "stochOverBought", "stochOverSold", "offset", "willOverSold", "willOverBought",
                   "nb pair", 'final_wallet mean', 'final_wallet max', 'vs_hold_pct mean', 'vs_hold_pct max',
                   'global_win_rate mean', 'global_win_rate max', 'total_trades mean', 'total_trades max']
    df_store_final_resutls = pd.DataFrame(columns=lst_columns)
    for sl in lst_stop_loss:
        for stochOverBought in lst_stochOverBought:
            for stochOverSold in lst_stochOverSold:
                for offset in lst_offset:
                    for willOverSold in lst_willOverSold:
                        for willOverBought in lst_willOverBought:
                            lst_row = []
                            df_tmp = df_final_results.copy()
                            df_tmp.drop(df_tmp[df_tmp['stop_loss'] != sl].index, inplace=True)
                            lst_row.append(sl)
                            df_tmp.drop(df_tmp[df_tmp['stochOverBought'] != stochOverBought].index, inplace=True)
                            lst_row.append(stochOverBought)
                            df_tmp.drop(df_tmp[df_tmp['stochOverSold'] != stochOverSold].index, inplace=True)
                            lst_row.append(stochOverSold)

                            df_tmp.drop(df_tmp[df_tmp['offset'] != offset].index, inplace=True)
                            lst_row.append(offset)

                            df_tmp.drop(df_tmp[df_tmp['willOverSold'] != willOverSold].index, inplace=True)
                            lst_row.append(willOverSold)
                            df_tmp.drop(df_tmp[df_tmp['willOverBought'] != willOverBought].index, inplace=True)
                            lst_row.append(willOverBought)

                            print('===========================')
                            print('sl: ', sl,
                                  " stochOverBought: ", stochOverBought,
                                  " stochOverSold: ", stochOverSold,
                                  " offset: ", offset,
                                  " willOverSold: ", willOverSold,
                                  " willOverBought: ", willOverBought
                                  )
                            print('nb pair: ', len(df_tmp))
                            lst_row.append(len(df_tmp))
                            print('final_wallet mean: ', df_tmp[['final_wallet']].mean().values[0])
                            lst_row.append(df_tmp[['final_wallet']].mean().values[0])
                            print('final_wallet max: ', max(df_tmp['final_wallet'].to_list()))
                            lst_row.append(max(df_tmp['final_wallet'].to_list()))
                            print('vs_hold_pct mean: ', df_tmp[['vs_hold_pct']].mean().values[0])
                            lst_row.append(df_tmp[['vs_hold_pct']].mean().values[0])
                            print('vs_hold_pct max: ', max(df_tmp['vs_hold_pct'].to_list()))
                            lst_row.append(max(df_tmp['vs_hold_pct'].to_list()))
                            print('global_win_rate mean: ', df_tmp[['global_win_rate']].mean().values[0])
                            lst_row.append(df_tmp[['global_win_rate']].mean().values[0])
                            print('global_win_rate max: ', max(df_tmp['global_win_rate'].to_list()))
                            lst_row.append(max(df_tmp['global_win_rate'].to_list()))
                            print('total_trades mean: ', df_tmp[['total_trades']].mean().values[0])
                            lst_row.append(df_tmp[['total_trades']].mean().values[0])
                            print('total_trades max: ', max(df_tmp['total_trades'].to_list()))
                            lst_row.append(max(df_tmp['total_trades'].to_list()))
                            print('list pairs: ', replace_in_list(df_tmp["pair"].to_list(), "/USDT", ""))

                            df_store_final_resutls.loc[len(df_store_final_resutls.index)] = lst_row

    df_store_final_resutls.to_csv("bollinger_reversion_final_global_results_filtered.csv")
    print('**************************')
    print('global list pairs: ', replace_in_list(list(set(df_final_results["pair"].to_list())), "/USDT", "")  )
    print('nb pairs: ', len(list(set(df_final_results["pair"].to_list()))))

    ENGAGED = False
    if ENGAGED:
        df_global_engaged['nb_engaged'] = df_global_engaged.sum(axis=1)
        df_global_engaged.to_csv("nb_engaged.csv")




