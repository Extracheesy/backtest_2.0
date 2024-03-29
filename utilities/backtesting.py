import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import conf.config

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def basic_single_asset_backtest(trades, days):
    df_trades = trades.copy()
    df_days = days.copy()
    
    df_days['evolution'] = df_days['wallet'].diff()
    df_days['daily_return'] = df_days['evolution']/df_days['wallet'].shift(1)
    
    df_trades['trade_result'] = df_trades["close_trade_size"] - df_trades["open_trade_size"] - df_trades["open_fee"]
    df_trades['trade_result_pct'] = df_trades['trade_result']/df_trades["open_trade_size"]
    df_trades['trade_result_pct_wallet'] = df_trades['trade_result']/(df_trades["wallet"]+df_trades["trade_result"])
    
    df_trades['wallet_ath'] = df_trades['wallet'].cummax()
    df_trades['drawdown'] = df_trades['wallet_ath'] - df_trades['wallet']
    df_trades['drawdown_pct'] = df_trades['drawdown'] / df_trades['wallet_ath']
    df_days['wallet_ath'] = df_days['wallet'].cummax()
    df_days['drawdown'] = df_days['wallet_ath'] - df_days['wallet']
    df_days['drawdown_pct'] = df_days['drawdown'] / df_days['wallet_ath']
    
    good_trades = df_trades.loc[df_trades['trade_result'] > 0]
    
    initial_wallet = df_days.iloc[0]["wallet"]
    total_trades = len(df_trades)
    total_good_trades = len(good_trades)
    avg_profit = df_trades['trade_result_pct'].mean()   
    global_win_rate = total_good_trades / total_trades
    max_trades_drawdown = df_trades['drawdown_pct'].max()
    max_days_drawdown = df_days['drawdown_pct'].max()
    final_wallet = df_days.iloc[-1]['wallet']
    buy_and_hold_pct = (df_days.iloc[-1]['price'] - df_days.iloc[0]['price']) / df_days.iloc[0]['price']
    buy_and_hold_wallet = initial_wallet + initial_wallet * buy_and_hold_pct
    vs_hold_pct = (final_wallet - buy_and_hold_wallet)/buy_and_hold_wallet
    vs_usd_pct = (final_wallet - initial_wallet)/initial_wallet
    sharpe_ratio = (365**0.5)*(df_days['daily_return'].mean()/df_days['daily_return'].std())
    total_fees = df_trades['open_fee'].sum() + df_trades['close_fee'].sum()
    
    best_trade = df_trades['trade_result_pct'].max()
    best_trade_date1 =  str(df_trades.loc[df_trades['trade_result_pct'] == best_trade].iloc[0]['open_date'])
    best_trade_date2 =  str(df_trades.loc[df_trades['trade_result_pct'] == best_trade].iloc[0]['close_date'])
    worst_trade = df_trades['trade_result_pct'].min()
    worst_trade_date1 =  str(df_trades.loc[df_trades['trade_result_pct'] == worst_trade].iloc[0]['open_date'])
    worst_trade_date2 =  str(df_trades.loc[df_trades['trade_result_pct'] == worst_trade].iloc[0]['close_date'])
    
    print("Period: [{}] -> [{}]".format(df_days.iloc[0]["day"], df_days.iloc[-1]["day"]))
    print("Initial wallet: {} $".format(round(initial_wallet,2)))
    
    print("\n--- General Information ---")
    print("Final wallet: {} $".format(round(final_wallet,2)))
    print("Performance vs US dollar: {} %".format(round(vs_usd_pct*100,2)))
    print("Sharpe Ratio: {}".format(round(sharpe_ratio,2)))
    print("Worst Drawdown T|D: -{}% | -{}%".format(round(max_trades_drawdown*100, 2), round(max_days_drawdown*100, 2)))
    print("Buy and hold performance: {} %".format(round(buy_and_hold_pct*100,2)))
    print("Performance vs buy and hold: {} %".format(round(vs_hold_pct*100,2)))
    print("Total trades on the period: {}".format(total_trades))
    print("Global Win rate: {} %".format(round(global_win_rate*100, 2)))
    print("Average Profit: {} %".format(round(avg_profit*100, 2)))
    print("Total fees paid {}$".format(round(total_fees, 2)))
    
    print("\nBest trades: +{} % the {} -> {}".format(round(best_trade*100, 2), best_trade_date1, best_trade_date2))
    print("Worst trades: {} % the {} -> {}".format(round(worst_trade*100, 2), worst_trade_date1, worst_trade_date2))

    return df_trades, df_days


def basic_single_asset_backtest_with_df(trades, days):
    df_trades = trades.copy()
    df_days = days.copy()

    lst_results = []
    lst_id_results = []

    df_days['evolution'] = df_days['wallet'].diff()
    df_days['daily_return'] = df_days['evolution'] / df_days['wallet'].shift(1)

    df_trades['trade_result'] = df_trades["close_trade_size"] - df_trades["open_trade_size"] - df_trades["open_fee"]
    df_trades['trade_result_pct'] = df_trades['trade_result'] / df_trades["open_trade_size"]
    df_trades['trade_result_pct_wallet'] = df_trades['trade_result'] / (df_trades["wallet"] + df_trades["trade_result"])

    df_trades['wallet_ath'] = df_trades['wallet'].cummax()
    df_trades['drawdown'] = df_trades['wallet_ath'] - df_trades['wallet']
    df_trades['drawdown_pct'] = df_trades['drawdown'] / df_trades['wallet_ath']
    df_days['wallet_ath'] = df_days['wallet'].cummax()
    df_days['drawdown'] = df_days['wallet_ath'] - df_days['wallet']
    df_days['drawdown_pct'] = df_days['drawdown'] / df_days['wallet_ath']

    good_trades = df_trades.loc[df_trades['trade_result'] > 0]

    initial_wallet = df_days.iloc[0]["wallet"]
    total_trades = len(df_trades)
    total_good_trades = len(good_trades)
    avg_profit = df_trades['trade_result_pct'].mean()
    global_win_rate = total_good_trades / total_trades
    max_trades_drawdown = df_trades['drawdown_pct'].max()
    max_days_drawdown = df_days['drawdown_pct'].max()
    final_wallet = df_days.iloc[-1]['wallet']
    buy_and_hold_pct = (df_days.iloc[-1]['price'] - df_days.iloc[0]['price']) / df_days.iloc[0]['price']
    buy_and_hold_wallet = initial_wallet + initial_wallet * buy_and_hold_pct
    try:
        vs_hold_pct = (final_wallet - buy_and_hold_wallet) / buy_and_hold_wallet
    except:
        vs_hold_pct = 0
        print("print exeception raised: vs_hold_pct = (final_wallet - buy_and_hold_wallet) / buy_and_hold_wallet")

    vs_usd_pct = (final_wallet - initial_wallet) / initial_wallet
    try:
        sharpe_ratio = (365 ** 0.5) * (df_days['daily_return'].mean() / df_days['daily_return'].std())
    except:
        sharpe_ratio = 0
        print("print exeception 0 raised: sharpe_ratio = (365 ** 0.5) * (df_days['daily_return'].mean() / df_days['daily_return'].std())")

    total_fees = df_trades['open_fee'].sum() + df_trades['close_fee'].sum()

    best_trade = df_trades['trade_result_pct'].max()
    best_trade_date1 = str(df_trades.loc[df_trades['trade_result_pct'] == best_trade].iloc[0]['open_date'])
    best_trade_date2 = str(df_trades.loc[df_trades['trade_result_pct'] == best_trade].iloc[0]['close_date'])
    worst_trade = df_trades['trade_result_pct'].min()
    worst_trade_date1 = str(df_trades.loc[df_trades['trade_result_pct'] == worst_trade].iloc[0]['open_date'])
    worst_trade_date2 = str(df_trades.loc[df_trades['trade_result_pct'] == worst_trade].iloc[0]['close_date'])

    if conf.config.PRINT_OUT:
        print("Period: [{}] -> [{}]".format(df_days.iloc[0]["day"], df_days.iloc[-1]["day"]))
        print("Initial wallet: {} $".format(round(initial_wallet, 2)))
        print("\n--- General Information ---")
        print("Final wallet: {} $".format(round(final_wallet, 2)))
        print("Performance vs US dollar: {} %".format(round(vs_usd_pct * 100, 2)))
        print("Sharpe Ratio: {}".format(round(sharpe_ratio, 2)))
        print("Worst Drawdown T|D: -{}% | -{}%".format(round(max_trades_drawdown * 100, 2),
                                                       round(max_days_drawdown * 100, 2)))
        print("Buy and hold performance: {} %".format(round(buy_and_hold_pct * 100, 2)))
        print("Performance vs buy and hold: {} %".format(round(vs_hold_pct * 100, 2)))
        print("Total trades on the period: {}".format(total_trades))
        print("Global Win rate: {} %".format(round(global_win_rate * 100, 2)))
        print("Average Profit: {} %".format(round(avg_profit * 100, 2)))
        print("Total fees paid {}$".format(round(total_fees, 2)))
        print("\nBest trades: +{} % the {} -> {}".format(round(best_trade * 100, 2), best_trade_date1, best_trade_date2))
        print("Worst trades: {} % the {} -> {}".format(round(worst_trade * 100, 2), worst_trade_date1, worst_trade_date2))

    lst_id_results.append("initial_wallet")
    lst_results.append(round(initial_wallet, 2))
    lst_id_results.append("final_wallet")
    lst_results.append(round(final_wallet, 2))
    lst_id_results.append("vs_usd_pct")
    lst_results.append(round(vs_usd_pct, 2))
    lst_id_results.append("sharpe_ratio")
    lst_results.append(round(sharpe_ratio, 2))
    lst_id_results.append("max_trades_drawdown")
    lst_results.append(round(max_trades_drawdown, 2))
    lst_id_results.append("max_days_drawdown")
    lst_results.append(round(max_days_drawdown, 2))
    lst_id_results.append("buy_and_hold_pct")
    lst_results.append(round(buy_and_hold_pct, 2))
    lst_id_results.append("vs_hold_pct")
    lst_results.append(round(vs_hold_pct, 2))
    lst_id_results.append("total_trades")
    lst_results.append(round(total_trades, 2))
    lst_id_results.append("global_win_rate")
    lst_results.append(round(global_win_rate, 2))
    lst_id_results.append("avg_profit")
    lst_results.append(round(avg_profit, 2))
    lst_id_results.append("total_fees")
    lst_results.append(round(total_fees, 2))

    df_result = pd.DataFrame(columns=lst_id_results)
    df_result.loc[len(df_result)] = lst_results
    return df_trades, df_days, df_result


def basic_multi_asset_backtest(trades, days):
    df_trades = trades.copy()
    df_days = days.copy()
    
    df_days['evolution'] = df_days['wallet'].diff()
    df_days['daily_return'] = df_days['evolution']/df_days['wallet'].shift(1)

    df_trades = df_trades.copy()
    df_trades['trade_result'] = df_trades["close_trade_size"] - df_trades["open_trade_size"] - df_trades["open_fee"] - df_trades["close_fee"]
    df_trades['trade_result_pct'] = df_trades['trade_result']/(df_trades["open_trade_size"] + df_trades["open_fee"])
    df_trades['trade_result_pct_wallet'] = df_trades['trade_result']/(df_trades["wallet"]+df_trades["trade_result"])
    good_trades = df_trades.loc[df_trades['trade_result_pct'] > 0]
    
    df_trades['wallet_ath'] = df_trades['wallet'].cummax()
    df_trades['drawdown'] = df_trades['wallet_ath'] - df_trades['wallet']
    df_trades['drawdown_pct'] = df_trades['drawdown'] / df_trades['wallet_ath']
    df_days['wallet_ath'] = df_days['wallet'].cummax()
    df_days['drawdown'] = df_days['wallet_ath'] - df_days['wallet']
    df_days['drawdown_pct'] = df_days['drawdown'] / df_days['wallet_ath']
    
    good_trades = df_trades.loc[df_trades['trade_result'] > 0]
    
    total_pair_traded = df_trades['pair'].nunique()
    initial_wallet = df_days.iloc[0]["wallet"]
    total_trades = len(df_trades)
    total_good_trades = len(good_trades)
    avg_profit = df_trades['trade_result_pct'].mean()   
    global_win_rate = total_good_trades / total_trades
    max_trades_drawdown = df_trades['drawdown_pct'].max()
    max_days_drawdown = df_days['drawdown_pct'].max()
    final_wallet = df_days.iloc[-1]['wallet']
    buy_and_hold_pct = (df_days.iloc[-1]['price'] - df_days.iloc[0]['price']) / df_days.iloc[0]['price']
    buy_and_hold_wallet = initial_wallet + initial_wallet * buy_and_hold_pct
    vs_hold_pct = (final_wallet - buy_and_hold_wallet)/buy_and_hold_wallet
    vs_usd_pct = (final_wallet - initial_wallet)/initial_wallet
    try:
        sharpe_ratio = (365**0.5)*(df_days['daily_return'].mean()/df_days['daily_return'].std())
    except:
        sharpe_ratio = 0
        print("print exeception 1 raised: sharpe_ratio = (365**0.5)*(df_days['daily_return'].mean()/df_days['daily_return'].std())")
    
    print("Period: [{}] -> [{}]".format(df_days.iloc[0]["day"], df_days.iloc[-1]["day"]))
    print("Initial wallet: {} $".format(round(initial_wallet,2)))
    print("Trades on {} pairs".format(total_pair_traded))
    
    print("\n--- General Information ---")
    print("Final wallet: {} $".format(round(final_wallet,2)))
    print("Performance vs US dollar: {} %".format(round(vs_usd_pct*100,2)))
    print("Sharpe Ratio: {}".format(round(sharpe_ratio,2)))
    print("Worst Drawdown T|D: -{}% | -{}%".format(round(max_trades_drawdown*100, 2), round(max_days_drawdown*100, 2)))
    print("Buy and hold performance: {} %".format(round(buy_and_hold_pct*100,2)))
    print("Performance vs buy and hold: {} %".format(round(vs_hold_pct*100,2)))
    print("Total trades on the period: {}".format(total_trades))
    print("Global Win rate: {} %".format(round(global_win_rate*100, 2)))
    print("Average Profit: {} %".format(round(avg_profit*100, 2)))
    
    print("\n----- Pair Result -----")
    print('-' * 95)
    print('{:<6s}{:>10s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}'.format(
                "Trades","Pair","Sum-result","Mean-trade","Worst-trade","Best-trade","Win-rate"
                ))
    print('-' * 95)
    for pair in df_trades["pair"].unique():
        df_pair = df_trades.loc[df_trades["pair"] == pair]
        pair_total_trades = len(df_pair)
        pair_good_trades = len(df_pair.loc[df_pair["trade_result"] > 0])
        pair_worst_trade = str(round(df_pair["trade_result_pct"].min() * 100, 2))+' %'
        pair_best_trade = str(round(df_pair["trade_result_pct"].max() * 100, 2))+' %'
        pair_win_rate = str(round((pair_good_trades / pair_total_trades) * 100, 2))+' %'
        pair_sum_result = str(round(df_pair["trade_result_pct"].sum() * 100, 2))+' %'
        pair_avg_result = str(round(df_pair["trade_result_pct"].mean() * 100, 2))+' %'
        print('{:<6d}{:>10s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}'.format(
                            pair_total_trades,pair,pair_sum_result,pair_avg_result,pair_worst_trade,pair_best_trade,pair_win_rate
                        ))
    
    return df_trades, df_days

def plot_sharpe_evolution(df_days):
    df_days_copy = df_days.copy()
    df_days_copy['evolution'] = df_days_copy['wallet'].diff()
    df_days_copy['daily_return'] = df_days_copy['evolution']/df_days_copy['wallet'].shift(1)

    df_days_copy['mean'] = df_days_copy['daily_return'].rolling(365).mean()
    df_days_copy['std'] = df_days_copy['daily_return'].rolling(365).std()
    df_days_copy['sharpe'] = (365**0.5)*(df_days_copy['mean']/df_days_copy['std'])
    df_days_copy['sharpe'].plot(figsize=(18, 9))

def plot_df_data(df):
    df_cp = df.copy()
    df_plot = pd.DataFrame(index=df_cp.index)
    df_plot["close"] = df_cp["close"]
    df_plot["bb_lowerband"] = df_cp["bb_lowerband"]
    df_plot["bb_lowerband_0985"] = df_cp["bb_lowerband_0985"]


    df_plot.plot()

    plt.show()

    df_plot.drop(columns=['close', 'bb_lowerband', 'bb_lowerband_0985'])

    df_plot["volume"] = df_cp["volume"]
    df_plot["volume_rolling"] = df_cp["volume_rolling"]

    df_plot.plot()

    plt.show()

    fig, ax_left = plt.subplots(figsize=(15, 20), nrows=2, ncols=1)

    ax_left[0].title.set_text("close vs bb_lowerband vs bb_lowerband_0985")
    ax_left[0].plot(df_plot['close'], color='royalblue', lw=1)
    ax_left[0].plot(df_plot['bb_lowerband'], color='red', lw=1)
    ax_left[0].plot(df_plot['bb_lowerband_0985'], color='green', lw=1)


    ax_left[1].title.set_text("volume vs volume_rolling")
    ax_left[1].plot(df_plot['volume'], color='royalblue', lw=1)
    ax_left[1].plot(df_plot['volume_rolling'], color='red', lw=1)

    plt.show()

def plot_wallet_vs_asset(df_days, log=False):
    days = df_days.copy()
    # print("-- Plotting equity vs asset and drawdown --")
    fig, ax_left = plt.subplots(figsize=(15, 20), nrows=4, ncols=1)

    ax_left[0].title.set_text("Strategy equity curve")
    ax_left[0].plot(days['wallet'], color='royalblue', lw=1)
    if log:
        ax_left[0].set_yscale('log')
    ax_left[0].fill_between(days['wallet'].index, days['wallet'], alpha=0.2, color='royalblue')
    ax_left[0].axhline(y=days.iloc[0]['wallet'], color='black', alpha=0.3)
    ax_left[0].legend(['Wallet evolution (equity)'], loc ="upper left")

    ax_left[1].title.set_text("Base currency evolution")
    ax_left[1].plot(days['price'], color='sandybrown', lw=1)
    if log:
        ax_left[1].set_yscale('log')
    ax_left[1].fill_between(days['price'].index, days['price'], alpha=0.2, color='sandybrown')
    ax_left[1].axhline(y=days.iloc[0]['price'], color='black', alpha=0.3)
    ax_left[1].legend(['Asset evolution'], loc ="upper left")

    ax_left[2].title.set_text("Drawdown curve")
    ax_left[2].plot(-days['drawdown_pct']*100, color='indianred', lw=1)
    ax_left[2].fill_between(days['drawdown_pct'].index, -days['drawdown_pct']*100, alpha=0.2, color='indianred')
    ax_left[2].axhline(y=0, color='black', alpha=0.3)
    ax_left[2].legend(['Drawdown in %'], loc ="lower left")

    ax_right = ax_left[3].twinx()
    if log:
        ax_left[3].set_yscale('log')
        ax_right.set_yscale('log')

    ax_left[3].title.set_text("Wallet VS Asset (not on the same scale)")
    ax_left[3].set_yticks([])
    ax_right.set_yticks([])
    ax_left[3].plot(days['wallet'], color='royalblue', lw=1)
    ax_right.plot(days['price'], color='sandybrown', lw=1)
    ax_left[3].legend(['Wallet evolution (equity)'], loc ="lower right")
    ax_right.legend(['Asset evolution'], loc ="upper left")

    plt.show()
    
def get_metrics(df_trades, df_days):
    df_days_copy = df_days.copy()
    df_days_copy['evolution'] = df_days_copy['wallet'].diff()
    df_days_copy['daily_return'] = df_days_copy['evolution']/df_days_copy['wallet'].shift(1)
    try:
        sharpe_ratio = (365**0.5)*(df_days_copy['daily_return'].mean()/df_days_copy['daily_return'].std())
    except:
        sharpe_ratio = 0
        print("print exeception 2 raised: sharpe_ratio = (365**0.5)*(df_days_copy['daily_return'].mean()/df_days_copy['daily_return'].std())")

    df_days_copy['wallet_ath'] = df_days_copy['wallet'].cummax()
    df_days_copy['drawdown'] = df_days_copy['wallet_ath'] - df_days_copy['wallet']
    df_days_copy['drawdown_pct'] = df_days_copy['drawdown'] / df_days_copy['wallet_ath']
    max_drawdown = -df_days_copy['drawdown_pct'].max() * 100
    
    df_trades_copy = df_trades.copy()
    df_trades_copy['trade_result'] = df_trades_copy["close_trade_size"] - df_trades_copy["open_trade_size"] - df_trades_copy["open_fee"] - df_trades_copy["close_fee"]
    df_trades_copy['trade_result_pct'] = df_trades_copy['trade_result']/df_trades_copy["open_trade_size"]
    df_trades_copy['trade_result_pct_wallet'] = df_trades_copy['trade_result']/(df_trades_copy["wallet"]+df_trades_copy["trade_result"])
    good_trades = df_trades_copy.loc[df_trades_copy['trade_result_pct'] > 0]
    win_rate = len(good_trades) / len(df_trades)
    avg_profit = df_trades_copy['trade_result_pct'].mean()
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "total_trades": len(df_trades_copy),
        "max_drawdown": max_drawdown
    }
    
def get_n_columns(df, columns, n=1):
    dt = df.copy()
    for col in columns:
        dt["n"+str(n)+"_"+col] = dt[col].shift(n)
    return dt

def plot_bar_by_month(df_days):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    custom_palette = {}
    
    last_month = int(df_days.iloc[-1]['day'].month)
    last_year = int(df_days.iloc[-1]['day'].year)
    
    current_month = int(df_days.iloc[0]['day'].month)
    current_year = int(df_days.iloc[0]['day'].year)
    current_year_array = []
    while current_year != last_year or current_month-1 != last_month:
        date_string = str(current_year) + "-" + str(current_month)
        
        monthly_perf = (df_days.loc[date_string]['wallet'].iloc[-1] - df_days.loc[date_string]['wallet'].iloc[0]) / df_days.loc[date_string]['wallet'].iloc[0]
        monthly_row = {
            'date': str(datetime.date(1900, current_month, 1).strftime('%B')),
            'result': round(monthly_perf*100)
        }
        if monthly_row["result"] >= 0:
            custom_palette[str(datetime.date(1900, current_month, 1).strftime('%B'))] = 'g'
        else:
            custom_palette[str(datetime.date(1900, current_month, 1).strftime('%B'))] = 'r'
        current_year_array.append(monthly_row)
        # print(monthly_perf*100) 
        if ((current_month == 12) or (current_month == last_month and current_year == last_year)):
            current_df = pd.DataFrame(current_year_array)
            # print(current_df)
            g = sns.barplot(data=current_df,x='date',y='result', palette=custom_palette)
            for index, row in current_df.iterrows():
                if row.result >= 0:
                    g.text(row.name,row.result, '+'+str(round(row.result))+'%', color='black', ha="center", va="bottom")
                else:
                    g.text(row.name,row.result, '-'+str(round(row.result))+'%', color='black', ha="center", va="top")
            g.set_title(str(current_year) + ' performance in %')
            g.set(xlabel=current_year, ylabel='performance %')
            
            year_result = (df_days.loc[str(current_year)]['wallet'].iloc[-1] - df_days.loc[str(current_year)]['wallet'].iloc[0]) / df_days.loc[str(current_year)]['wallet'].iloc[0]
            print("----- " + str(current_year) +" Cumulative Performances: " + str(round(year_result*100,2)) + "% -----")
            plt.show()

            current_year_array = []
        
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1