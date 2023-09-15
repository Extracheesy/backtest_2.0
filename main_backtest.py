# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from datetime import datetime, timedelta
import conf.config
import warnings

from src.mean_reversion import MeanReversion
from src.mean_bol_trend import MeanBolTrend
from src.rsi_bb_sma import RSI_BB_SMA
from src.bol_trend_live import BolTrendLive
from src.hull_suite import HullSuite
from src.envelope import Envelope
import src.backtest
from src.benchmark import Benchmark
from utilities.utils import create_directory, clean_df_columns, get_lst_intervals_name, clean_df, get_dir_strating_with, copy_files_to_target_dir, rm_dir, merge_csv, get_lst_dir_strating_with
from src.bigwill import BigWill
from src.cluc_may import ClucMay
from src.scalping_engulfing import ScalpingEngulfing
from src.analyse_pair import AnalysePair
from src.cryptobot_indicators import TechnicalAnalysis
from src.filter import filter_df_from_column_values, filter_and_benchmark_global_results

from src.slope_is_dope import SlopeIsDope

import numpy as np

import os, sys
import glob
import asyncio
import conf.config
from src.crypto_data import ExchangeDataManager



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) >= 2:
        if sys.argv[1] == "--COLAB":
            conf.config.COLAB = True
            results_path = conf.config.COLAB_DIR_ROOT + "./results/"
    else:
        lst_dir_results = get_lst_dir_strating_with('./', "results_")
        # Extract the numeric suffixes and convert them to integers
        suffixes = [int(entry.split("_")[1]) for entry in lst_dir_results]
        if len(suffixes) == 0:
            max_suffix = 1
        else:
            # Find the maximum value among the suffixes
            max_suffix = max(suffixes) + 1
        if max_suffix < 10:
            str_max_suffix = "0" + str(max_suffix)
        else:
            str_max_suffix = str(max_suffix)
        results_path = "./results_" + str_max_suffix + "/"

    if conf.config.NO_WARNINGS:
        # To filter out all warnings and hide them
        warnings.filterwarnings("ignore")

    create_directory(results_path)

    lst_symbol = src.backtest.get_lst_symbols(conf.config.symbol)
    lst_pair = src.backtest.get_lst_pair(lst_symbol)

    run_start = datetime.now()

    if conf.config.GET_DATA:
        print("************** RUN GET DATA **************")
        tf = conf.config.tf
        start = conf.config.start
        src.backtest.get_ohlvc(tf, start, lst_pair)
        print('elapsed time: ', datetime.now() - run_start)
        print("************** GET DATA COMPLETED **************")

    if conf.config.VOLATILITY_ANALYSE:
        src.backtest.analyse_envelope_volatility(conf.config.lst_offset, lst_pair)

    df_global_engaged = pd.DataFrame()

    lst_strategy = conf.config.lst_strategy
    lst_type = conf.config.lst_type
    if conf.config.RUN_ON_INTERVALS:
        lst_filter_start = get_lst_intervals_name(conf.config.INTERVALS, "INTRV")
    else:
        lst_filter_start = conf.config.lst_filter_start
    tf = conf.config.tf


    if conf.config.RUN_BACKTEST:
        print("************** RUN BACKTEST **************")
        df_final_results = src.backtest.run_main_backtest(lst_strategy, lst_pair, lst_type, tf, lst_filter_start)
        print('elapsed time: ', datetime.now() - run_start)
        print("************** BACKTEST COMPLETED **************")
    else:
        df_final_results = pd.read_csv(results_path + "final_global_results_no_filter.csv")
        df_final_results = clean_df_columns(df_final_results)

    if conf.config.RUN_BENCHMARK:
        print("************** RUN BENCHMARK **************")
        benchmark = Benchmark(df_final_results)
        benchmark.run_benchmark()
        benchmark.export_benchmark_strategy(results_path)

        rows = len(df_final_results)
        print("=> final_global_results_no_filter.csv")
        df_final_results.to_csv(results_path + "final_global_results_no_filter.csv")
        lst_df_resullts_fileterd = []
        df_tmp_df_final_results_final_wallet = df_final_results.copy()
        df_tmp_df_final_results_final_wallet.drop(df_tmp_df_final_results_final_wallet[df_tmp_df_final_results_final_wallet['final_wallet'] <= 1000].index, inplace=True)
        df_tmp_df_final_results_final_wallet.reset_index(drop=True, inplace=True)
        df_tmp_df_final_results_final_wallet.to_csv(results_path + "final_global_results_filtered_final_wallet.csv")
        lst_df_resullts_fileterd.append(df_tmp_df_final_results_final_wallet)

        df_tmp_df_final_results_vs_hold_pct = df_final_results.copy()
        df_tmp_df_final_results_vs_hold_pct.drop(df_tmp_df_final_results_vs_hold_pct[df_tmp_df_final_results_vs_hold_pct['vs_hold_pct'] < 0.1].index, inplace=True)
        df_tmp_df_final_results_vs_hold_pct.reset_index(drop=True, inplace=True)
        df_tmp_df_final_results_vs_hold_pct.to_csv(results_path + "final_global_results_filtered_vs_hold_pct.csv")
        lst_df_resullts_fileterd.append(df_tmp_df_final_results_vs_hold_pct)

        df_tmp_df_final_results_sharpe_ratio = df_final_results.copy()
        df_tmp_df_final_results_sharpe_ratio.drop(df_tmp_df_final_results_sharpe_ratio[df_tmp_df_final_results_sharpe_ratio['sharpe_ratio'] < 1.0].index, inplace=True)
        df_tmp_df_final_results_sharpe_ratio.reset_index(drop=True, inplace=True)
        df_tmp_df_final_results_sharpe_ratio.to_csv(results_path + "final_global_results_filtered_sharpe_ratio.csv")
        lst_df_resullts_fileterd.append(df_tmp_df_final_results_sharpe_ratio)

        if False:
            df_tmp_df_final_results_global_win_rate = df_final_results.copy()
            df_tmp_df_final_results_global_win_rate.drop(df_tmp_df_final_results_global_win_rate[df_tmp_df_final_results_global_win_rate['global_win_rate'] < 0.5].index, inplace=True)
            df_tmp_df_final_results_global_win_rate.reset_index(drop=True, inplace=True)
            lst_df_resullts_fileterd.append(df_tmp_df_final_results_global_win_rate)

        df_final_results_filtered = pd.concat(lst_df_resullts_fileterd,
                                              ignore_index=True, sort=False)
        df_final_results_filtered.reset_index(drop=True, inplace=True)
        rows_filtered = len(df_final_results_filtered)
        df_final_results_filtered.drop_duplicates(inplace=True)
        duplicates_filtered = rows_filtered - len(df_final_results_filtered)
        print("fileted rows: ", rows_filtered, " filtered dropped duplicates rows: ", duplicates_filtered, " rows")

        print("rows: ", rows, " filtered  rows: ", rows - len(df_final_results_filtered))
        print("=> final_global_results_filtered.csv")
        df_final_results_filtered.to_csv(results_path + "final_global_results_filtered.csv")
        print('elapsed time: ', datetime.now() - run_start)
        print("************** BENCHMARK COMPLETED **************")


    if conf.config.RUN_FILTER:
        print("************** RUN FILTER **************")

        # Get the current date
        current_date = datetime.now()
        # Format the year, day, and month as "YYDDMM"
        year_day_month_string = current_date.strftime("%y%d%m")

        df_param = pd.read_csv(results_path + "benchmark_transposed_final_wallet-" + year_day_month_string + ".csv", sep=';')
        df_param = clean_df_columns(df_param)
        # df_pairs = pd.read_csv(results_path + "benchmark_compare_pairs.csv")

        df_filtered = filter_df_from_column_values(df_param, 0.9) # 10%
        df_filtered = clean_df(df_filtered)
        df_filtered.to_csv(results_path + "benchmark_transposed_final_wallet_filterd-" + year_day_month_string + ".csv", sep=';')
        print('elapsed time: ', datetime.now() - run_start)
        print("************** FILTER COMPLETED **************")

    if conf.config.GLOBAL_FILTER:
        print("************** RUN GLOBAL FILTER **************")
        lst_results_dir = get_dir_strating_with('./', 'results_')
        rm_dir(conf.config.final_target_results)
        create_directory(conf.config.final_target_results)
        copy_files_to_target_dir(lst_results_dir, conf.config.final_target_results, "benchmark_transposed_final_wallet_filterd")
        merge_csv(conf.config.final_target_results, "benchmark_transposed_final_wallet_filterd", "merged_df_reults.csv")
        df_global_results, df_global_results_filtered_1, df_global_results_filtered_2 = filter_and_benchmark_global_results(conf.config.final_target_results, "merged_df_reults.csv")

        df_global_results = clean_df(df_global_results)
        df_global_results.to_csv(conf.config.final_target_results + "/" + "merged_df_results_analysed.csv", sep=";")

        df_global_results_filtered_1 = clean_df(df_global_results_filtered_1)
        df_global_results_filtered_1.to_csv(conf.config.final_target_results + "/" + "merged_df_results_analysed_filtered_1_best.csv", sep=";")

        df_global_results_filtered_2 = clean_df(df_global_results_filtered_2)
        df_global_results_filtered_2.to_csv(conf.config.final_target_results + "/" + "merged_df_results_analysed_filtered_2_best.csv", sep=";")
        print("************** RUN GLOBAL FILTER COMPLETED **************")

    print('final elapsed time: ', datetime.now() - run_start)

    exit()

    lst_columns = ['stop_loss',
                   "offset",
                   "bol_window", "bol_std",
                   "min_bol_spread", "long_ma_window",
                   "stochOverBought", "stochOverSold", "willOverSold", "willOverBought"
                   "nb pair", 'final_wallet mean', 'final_wallet max', 'vs_hold_pct mean', 'vs_hold_pct max',
                   'global_win_rate mean', 'global_win_rate max', 'total_trades mean', 'total_trades max']
    df_store_final_resutls = pd.DataFrame(columns=lst_columns)
    for sl in lst_stop_loss:
        for bol_window in lst_bol_window:
            for bol_std in lst_bol_std:
                for offset in lst_offset:
                    for min_bol_spread in lst_min_bol_spread:
                        for long_ma_window in lst_long_ma_window:
                            lst_row = []
                            df_tmp = df_final_results.copy()
                            df_tmp.drop(df_tmp[df_tmp['stop_loss'] != sl].index, inplace=True)
                            lst_row.append(sl)
                            df_tmp.drop(df_tmp[df_tmp['bol_window'] != bol_window].index, inplace=True)
                            lst_row.append(bol_window)
                            df_tmp.drop(df_tmp[df_tmp['bol_std'] != bol_std].index, inplace=True)
                            lst_row.append(bol_std)

                            df_tmp.drop(df_tmp[df_tmp['offset'] != offset].index, inplace=True)
                            lst_row.append(offset)

                            df_tmp.drop(df_tmp[df_tmp['min_bol_spread'] != min_bol_spread].index, inplace=True)
                            lst_row.append(min_bol_spread)
                            df_tmp.drop(df_tmp[df_tmp['long_ma_window'] != long_ma_window].index, inplace=True)
                            lst_row.append(long_ma_window)

                            if conf.config.PRINT_OUT:
                                print('===========================')
                                print('sl: ', sl,
                                      " bol_window: ", bol_window,
                                      " bol_std: ", bol_std,
                                      " offset: ", offset,
                                      " min_bol_spread: ", min_bol_spread,
                                      " long_ma_window: ", long_ma_window
                                      )
                                print('nb pair: ', len(df_tmp))
                                print('final_wallet mean: ', df_tmp[['final_wallet']].mean().values[0])
                                print('final_wallet max: ', max(df_tmp['final_wallet'].to_list()))
                                print('vs_hold_pct mean: ', df_tmp[['vs_hold_pct']].mean().values[0])
                                print('vs_hold_pct max: ', max(df_tmp['vs_hold_pct'].to_list()))
                                print('global_win_rate mean: ', df_tmp[['global_win_rate']].mean().values[0])
                                print('global_win_rate max: ', max(df_tmp['global_win_rate'].to_list()))
                                print('total_trades mean: ', df_tmp[['total_trades']].mean().values[0])
                                print('total_trades max: ', max(df_tmp['total_trades'].to_list()))
                                print('list pairs: ', src.backtest.replace_in_list(df_tmp["pair"].to_list(), "/USDT", ""))

                            lst_row.append(len(df_tmp))
                            lst_row.append(df_tmp[['final_wallet']].mean().values[0])
                            lst_row.append(max(df_tmp['final_wallet'].to_list()))
                            lst_row.append(df_tmp[['vs_hold_pct']].mean().values[0])
                            lst_row.append(max(df_tmp['vs_hold_pct'].to_list()))
                            lst_row.append(df_tmp[['global_win_rate']].mean().values[0])
                            lst_row.append(max(df_tmp['global_win_rate'].to_list()))
                            lst_row.append(df_tmp[['total_trades']].mean().values[0])
                            lst_row.append(max(df_tmp['total_trades'].to_list()))
                            df_store_final_resutls.loc[len(df_store_final_resutls.index)] = lst_row

    df_store_final_resutls.to_csv("bollinger_final_global_results_filtered.csv")
    if conf.config.PRINT_OUT:
        print('**************************')
        print('global list pairs: ', src.backtest.replace_in_list(list(set(df_final_results["pair"].to_list())), "/USDT", "")  )
        print('nb pairs: ', len(list(set(df_final_results["pair"].to_list()))))
    else:
        print('**************************')
        print('global list pairs: ', src.backtest.replace_in_list(list(set(df_final_results["pair"].to_list())), "/USDT", "")  )
        print('nb pairs: ', len(list(set(df_final_results["pair"].to_list()))))

    df_global_engaged['nb_engaged'] = df_global_engaged.sum(axis=1)
    filename = "mapping_overlap_engaged.csv"
    # df_global_engaged.to_csv(filename)
    # plot_engaged_overlap(filename)



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

