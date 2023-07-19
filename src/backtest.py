import pandas as pd
from utilities.get_data import get_historical_from_db
import os
import glob
import ccxt
import conf.config
from src.bol_trend import BolTrend
from src.bigwill import BigWill
from src.bollinger_reversion import BollingerReversion
import matplotlib.pyplot as plt

from utilities.backtesting import basic_single_asset_backtest_with_df, basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month

def replace_in_list(original_list, string_to_replace, replacement_string):
    modified_list = []
    for item in original_list:
        modified_item = item.replace(string_to_replace, replacement_string)
        modified_list.append(modified_item)
    return modified_list

def analyse_envelope_volatility(lst_offset, lst_pair):
    for offset in lst_offset:
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
    df_results.to_csv("envelope_final_results.csv")

def get_ohlvc(tf, start, lst_pair):
    try:
        exchange = ExchangeDataManager(exchange_name="binance", path_download="./database/exchanges")
        asyncio.run(exchange.download_data(lst_pair, [tf], start_date=start))
    except:
        print("download failure")


def get_lst_pair(lst_symbol):
    lst_pair = []
    for symbol in lst_symbol:
        lst_pair.append(symbol + "/USDT")
    return lst_pair

def get_lst_symbols(key):
    if key == 'ALL':
        return conf.config.lst_symbol_ALL
    if key == 'BITGET':
        return conf.config.lst_symbol_BITGET
    if key == 'BTC_ETH':
        return conf.config.lst_symbol_BTC_ETH
    if key == 'BTC':
        return conf.config.lst_symbol_BTC
    if key == 'ETH':
        return conf.config.lst_symbol_ETH

def run_main_backtest(lst_strategy, lst_pair, lst_type, tf, lst_filter_start):
    df_pair = pd.DataFrame(columns=['pair', 'df_pair'])
    for pair in lst_pair:
        df = get_historical_from_db(
            ccxt.binance(),
            pair,
            tf,
            path="./database/"
        )
        lst_pair_row = [pair, df]
        df_pair.loc[len(df_pair.index)] = lst_pair_row

    df_pair = df_pair.set_index('pair')
    df_backtest_results = pd.DataFrame()
    for filter_start in lst_filter_start:
        for strategy in lst_strategy:
            df_tmp = run_strategy_backtest(strategy, df_pair, lst_type, tf, filter_start)
            if len(df_backtest_results):
                df_backtest_results = df_tmp.copy()
            else:
                df_backtest_results = pd.concat([df_backtest_results, df_tmp], axis=0)
    return df_backtest_results

def plot_engaged_overlap(filename):
    # filename = 'nb_engaged.csv'
    dt_tmp_test = pd.read_csv(filename)

    dt_tmp_test['X'] = dt_tmp_test.index

    # Plot the 'Count' column as a bar chart
    dt_tmp_test.plot(x='X', y='nb_engaged', kind='bar')

    # Set labels and title
    plt.xlabel('nb symbol engaged with bigwill')
    plt.ylabel('Count')
    plt.title('Big Will')

    # Display the plot
    plt.show()

def get_best_performer(df, nb_performer, lst_performer):
    df_perf = df.copy()
    df_perf = df_perf.loc[:0]
    for col in lst_performer:
        df_tmp = df.copy()
        df_tmp = df_tmp.sort_values(by=[col], ascending=False)
        df_tmp.reset_index(drop=True, inplace=True)

        df_tmp = df_tmp.loc[:nb_performer]
        df_perf = pd.concat([df_perf, df_tmp], axis=0)

    df_perf.drop_duplicates(inplace=True)
    df_perf = df_perf.sort_values(by=["vs_hold_pct"], ascending=False)
    df_perf.reset_index(drop=True, inplace=True)
    return df_perf

def run_strategy(lst_param):
    lst_type = conf.config.lst_type
    filter_start = lst_param[0]
    strategy = lst_param[1]
    sl = lst_param[2]
    bol_window = lst_param[3]
    bol_std = lst_param[4]
    offset = lst_param[5]
    min_bol_spread = lst_param[6]
    long_ma_window = lst_param[7]
    stochOverBought = lst_param[8]
    stochOverSold = lst_param[9]
    willOverSold = lst_param[10]
    willOverBought = lst_param[11]
    pair = lst_param[12]
    df = lst_param[13]

    df_final_results = pd.DataFrame()
    if strategy == "bol_trend":
        try:
            strat = BolTrend(
                df=df,
                type=lst_type,
                bol_window=bol_window,
                bol_std=bol_std,
                min_bol_spread=min_bol_spread,
                long_ma_window=long_ma_window,
                stochOverBought=stochOverBought,
                stochOverSold=stochOverSold,
                willOverSold=willOverSold,
                willOverBought=willOverBought,
                SL=sl,
                TP=0
            )
        except:
            print("toto")
    elif strategy == "big_will":
        strat = BigWill(
            df=df,
            type=lst_type,
            stochOverBought=stochOverBought,
            stochOverSold=stochOverSold,
            willOverSold=willOverSold,
            willOverBought=willOverBought,
            SL=sl,
            TP=0
        )
    elif strategy == "bollinger_reversion":
        strat = BollingerReversion(
            df=df,
            type=lst_type,
            bol_window=bol_window,
            bol_std=bol_std,
            min_bol_spread=min_bol_spread,
            long_ma_window=long_ma_window,
            stochOverBought=stochOverBought,
            stochOverSold=stochOverSold,
            willOverSold=willOverSold,
            willOverBought=willOverBought,
            SL=sl,
            TP=0
        )
    strat.populate_indicators()
    strat.populate_buy_sell()

    bt_result = strat.run_backtest(initial_wallet=1000, leverage=5)

    if conf.config.ENGAGED_OVERLAP:
        if bt_result != None:
            df_engaged = strat.get_df_engaged(pair, bt_result)
            if len(df_global_engaged) == 0:
                df_global_engaged = df_engaged
            else:
                df_global_engaged[pair] = df_engaged[pair]

    if conf.config.PRINT_OUT:
        print("pair: ", pair, " offset: ", offset)
    if bt_result != None:
        df_trades, df_days, df_tmp = basic_single_asset_backtest_with_df(
            trades=bt_result['trades'], days=bt_result['days'])
    else:
        lst_columns = ["initial_wallet", "final_wallet", "vs_usd_pct",
                       "sharpe_ratio", "max_trades_drawdown",
                       "max_days_drawdown", "buy_and_hold_pct",
                       "vs_hold_pct", "total_trades", "global_win_rate",
                       "avg_profit", "total_fees"]
        lst_data_nul = [1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        df_tmp = pd.DataFrame(columns=lst_columns)
        df_tmp.loc[len(df_tmp)] = lst_data_nul

    df_tmp['start'] = filter_start
    df_tmp['strategy'] = strategy
    df_tmp['pair'] = pair
    df_tmp['stop_loss'] = sl
    df_tmp["offset"] = offset

    df_tmp["bol_window"] = bol_window
    df_tmp["bol_std"] = bol_std
    df_tmp["min_bol_spread"] = min_bol_spread
    df_tmp["long_ma_window"] = long_ma_window

    df_tmp["stochOverBought"] = stochOverBought
    df_tmp["stochOverSold"] = stochOverSold
    df_tmp["willOverSold"] = willOverSold
    df_tmp["willOverBought"] = willOverBought

    if len(df_final_results) == 0:
        df_final_results = df_tmp.copy()
    else:
        df_final_results = pd.concat([df_final_results, df_tmp],
                                     ignore_index=True, sort=False)

    return df_final_results

def run_strategy_backtest(strategy, df_pair, lst_type, tf, filter_start):
    print("strategy: ", strategy, " strart: ", filter_start)
    df_final_results = pd.DataFrame()

    lst_pair = df_pair.index.to_list()
    lst_pair = list(dict.fromkeys(lst_pair))

    lst_stop_loss = conf.config.dct_lst_param["lst_stop_loss"]
    lst_bol_window = conf.config.dct_lst_param["lst_bol_window"]
    lst_bol_std = conf.config.dct_lst_param["lst_bol_std"]
    lst_offset = conf.config.dct_lst_param["lst_offset"]
    lst_min_bol_spread = conf.config.dct_lst_param["lst_min_bol_spread"]
    lst_long_ma_window = conf.config.dct_lst_param["lst_long_ma_window"]

    lst_stochOverBought = conf.config.dct_lst_param["lst_stochOverBought"]
    lst_stochOverSold = conf.config.dct_lst_param["lst_stochOverSold"]
    lst_willOverSold = conf.config.dct_lst_param["lst_willOverSold"]
    lst_willOverBought = conf.config.dct_lst_param["lst_willOverBought"]

    if strategy == "bol_trend":
        lst_offset = [0]
        lst_stochOverBought = [0]
        lst_stochOverSold = [0]
        lst_willOverSold = [0]
        lst_willOverBought = [0]
    elif strategy == "big_will":
        lst_offset = [0]
        lst_bol_window = [0]
        lst_bol_std = [0]
        lst_min_bol_spread = [0]
        lst_long_ma_window = [0]
    elif strategy == "bollinger_reversion":
        lst_offset = [0]

    lst_of_lst_parameters = []
    for sl in lst_stop_loss:
        lst_parametest = [filter_start, strategy, sl]
        for bol_window in lst_bol_window:
            lst_parametest.append(bol_window)
            for bol_std in lst_bol_std:
                lst_parametest.append(bol_std)
                for offset in lst_offset:
                    lst_parametest.append(offset)
                    for min_bol_spread in lst_min_bol_spread:
                        lst_parametest.append(min_bol_spread)
                        for long_ma_window in lst_long_ma_window:
                            lst_parametest.append(long_ma_window)
                            for stochOverBought in lst_stochOverBought:
                                lst_parametest.append(stochOverBought)
                                for stochOverSold in lst_stochOverSold:
                                    lst_parametest.append(stochOverSold)
                                    for willOverSold in lst_willOverSold:
                                        lst_parametest.append(willOverSold)
                                        for willOverBought in lst_willOverBought:
                                            lst_parametest.append(willOverBought)
                                            for pair in lst_pair:
                                                lst_parametest_tmp = lst_parametest
                                                df = df_pair.at[pair, "df_pair"]
                                                df = df.loc[filter_start:]
                                                lst_parametest_tmp.append(pair)
                                                lst_parametest_tmp.append(df)
                                                lst_of_lst_parameters.append(lst_parametest_tmp)

    lst_df_results = list(map(run_strategy, lst_of_lst_parameters))
    df_results = pd.concat(lst_df_results, ignore_index=True, sort=False)
    return df_results

