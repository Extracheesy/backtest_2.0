import pandas as pd
import numpy as np
import conf.config
from itertools import product
from datetime import datetime, timedelta
import multiprocessing
from utilities.utils import move_column_to_first_position, remove_values_from_lst, clean_df

class Benchmark():
    def __init__(
        self,
        df,
    ):
        self.df = df
        move_column_to_first_position(self.df, "pair")
        move_column_to_first_position(self.df, "strategy")
        move_column_to_first_position(self.df, "start")
        if "initial_wallet" in df.columns.tolist():
            self.df.drop(columns=["initial_wallet"], inplace=True)
        self.df_performers = pd.DataFrame(columns=df.columns.tolist())
        self.df_benchmark_strategy = pd.DataFrame(columns=conf.config.LST_COLUMN_STRATEGY_BENCHMARK)
        self.df['start'] = self.df['start'].astype(str)
        self.lst_start_date = self.get_lst_from_df_column(self.df, 'start')
        self.lst_strategy = self.get_lst_from_df_column(self.df, 'strategy')
        self.lst_pair = self.get_lst_from_df_column(self.df, 'pair')
        self.df_transposed_final_wallet = self.transpose_df(self.df, "final_wallet")
        self.df_benchmark_parameters = pd.DataFrame(columns=conf.config.LST_COLUMN_PARAMETER_BENCHMARK + self.lst_pair)
        self.df_benchmark_pair_lst = pd.DataFrame(columns=conf.config.LST_HEADERS_PAIRS_BENCMARK)
        self.df_benchmark_pair_lst_compare = pd.DataFrame(columns=conf.config.LST_HEADERS_LST_PAIRS_COMPARE_BENCHMARK)
        self.df_benchmark_pair_lst_compare["pair"] = self.lst_pair
        self.current_date = datetime.now()
        self.year_day_month_string = self.current_date.strftime("%y%d%m")

    def run_benchmark(self):
        df = self.df.copy()

        df.sort_values(by=['final_wallet'], inplace=True , ascending=True)
        df['final_wallet'] = np.where(df['final_wallet'] < 0, 0, df['final_wallet'])
        self.df = df
        self.benchmark_strategy()
        self.benchmark_performers()

    def export_benchmark_strategy(self, path):
        self.df_benchmark_strategy = self.df_benchmark_strategy.round(2)
        self.df_benchmark_strategy = clean_df(self.df_benchmark_strategy)
        self.df_benchmark_strategy.to_csv(path + "benchmark_strategy.csv", sep=';')

        self.df_performers = self.df_performers.round(2)
        self.df_performers = clean_df(self.df_performers)
        self.df_performers.to_csv(path + "benchmark_performer.csv", sep=';')

        self.df_benchmark_parameters.to_csv(path + "benchmark_parameters.csv", sep=';')

        self.df_benchmark_pair_lst.to_csv(path + "benchmark_pairs.csv", sep=';')

        self.df_benchmark_pair_lst_compare.to_csv(path + "benchmark_compare_pairs.csv", sep=';')

        self.df_transposed_final_wallet = self.df_transposed_final_wallet.round(4)
        self.df_transposed_final_wallet = clean_df(self.df_transposed_final_wallet)



        self.df_transposed_final_wallet.to_csv(path + "benchmark_transposed_final_wallet-" + self.year_day_month_string + ".csv", sep=';')

    def get_lst_from_df_column(self,df, column):
        return list(dict.fromkeys(df[column].to_list()))

    def benchmark_strategy(self):
        for start_date in self.lst_start_date:
            df_strategy = self.df.copy()
            df_strategy.drop(df_strategy[df_strategy['start'] != start_date].index, inplace=True)
            df_strategy_tmp = df_strategy.copy()
            for strategy in self.lst_strategy:
                df_strategy = df_strategy_tmp.copy()
                df_strategy.drop(df_strategy[df_strategy['strategy'] != strategy].index, inplace=True)
                self.benchmark_strategy_analyse_values(start_date, strategy, df_strategy)
                self.benchmark_pair_analyse_values(start_date, strategy, df_strategy)
                self.benchmark_parameters_analyse(start_date, strategy, df_strategy)

        for start_date in self.lst_start_date:
            df_parameters = self.df_benchmark_parameters.copy()
            df_parameters.drop(df_parameters[df_parameters['start_date'] != start_date].index, inplace=True)
            df_parameters_tmp = df_parameters.copy()
            for strategy in self.lst_strategy:
                df_parameters = df_parameters_tmp.copy()
                df_parameters.drop(df_parameters[df_parameters['strategy'] != strategy].index, inplace=True)
                self.benchmark_pair_analyse(start_date, strategy, df_parameters)

        self.df_benchmark_pair_lst.drop(self.df_benchmark_pair_lst[self.df_benchmark_pair_lst['score'] == 0].index, inplace=True)
        self.df_benchmark_pair_lst_compare = self.df_benchmark_pair_lst_compare.reindex(sorted(self.df_benchmark_pair_lst_compare.columns), axis=1)
        self.df_benchmark_pair_lst_compare = move_column_to_first_position(self.df_benchmark_pair_lst_compare, "pair")

        for strategy in self.lst_strategy:
            self.benchmark_pair_final_lst(strategy)

    def benchmark_performers(self):
        for start_date in self.lst_start_date:
            df_strategy = self.df.copy()
            df_strategy.drop(df_strategy[df_strategy['start'] != start_date].index, inplace=True)
            self.benchmark_analyse_performers(df_strategy)
            df_strategy_tmp = df_strategy.copy()
            for strategy in self.lst_strategy:
                df_strategy = df_strategy_tmp.copy()
                df_strategy.drop(df_strategy[df_strategy['strategy'] != strategy].index, inplace=True)
                self.benchmark_analyse_performers(df_strategy)
                df_pair_tmp = df_strategy.copy()
                for pair in self.lst_pair:
                    df_strategy = df_pair_tmp.copy()
                    df_strategy.drop(df_strategy[df_strategy['pair'] != pair].index, inplace=True)
                    self.benchmark_analyse_performers(df_strategy)

    def benchmark_analyse_performers(self, df):
        df_performer = df.copy()
        # Calculate the threshold for the top 20% values
        threshold = max(df_performer['final_wallet'].quantile(conf.config.QUANTILE), 1000)

        # Filter the DataFrame to keep only the top 10% values
        df_performer_wallet = df_performer[df_performer['final_wallet'] >= threshold]
        df_performer_wallet = df_performer_wallet[df_performer_wallet['final_wallet'] > 1000]

        # Calculate the threshold for the top 20% values
        threshold = max(df_performer['sharpe_ratio'].quantile(conf.config.QUANTILE), 1)
        df_performer_sharpe_ratio = df_performer[df_performer['sharpe_ratio'] >= threshold]
        df_performer_sharpe_ratio = df_performer_sharpe_ratio[df_performer_sharpe_ratio['final_wallet'] > 1000]

        self.df_performers = pd.concat([self.df_performers, df_performer_wallet, df_performer_sharpe_ratio])
        self.df_performers.reset_index(inplace=True, drop=True)
        self.df_performers.drop_duplicates(inplace=True)

        self.df_performers = self.df_performers[self.df_performers['vs_hold_pct'] >= 0]
        self.df_performers = self.df_performers[self.df_performers['total_trades'] > 0]
        self.df_performers.reset_index(inplace=True, drop=True)

    def get_percentage_over_threshold_in_df(self, df, column_to_check, threshold):
        total_rows = len(df)
        if total_rows == 0:
            return 0
        values_above_threshold = len(df[df[column_to_check] > threshold])
        percentage_above_threshold = (values_above_threshold / total_rows) * 100
        return percentage_above_threshold

    def benchmark_strategy_analyse_values(self, start_date, strategy, df):
        lst_results = [start_date, strategy, "-"]
        max_final_wallet = df['final_wallet'].max()
        min_final_wallet = df['final_wallet'].min()
        mean_final_wallet = df['final_wallet'].mean()
        percent_over_1000_final_wallet = self.get_percentage_over_threshold_in_df(df, 'final_wallet', 1000)
        percent_over_1500_final_wallet = self.get_percentage_over_threshold_in_df(df, 'final_wallet', 1500)
        percent_over_2000_final_wallet = self.get_percentage_over_threshold_in_df(df, 'final_wallet', 2000)
        percent_over_2500_final_wallet = self.get_percentage_over_threshold_in_df(df, 'final_wallet', 2500)
        percent_over_3500_final_wallet = self.get_percentage_over_threshold_in_df(df, 'final_wallet', 3000)
        lst_results.append(max_final_wallet)
        lst_results.append(min_final_wallet)
        lst_results.append(mean_final_wallet)
        lst_results.append(percent_over_1000_final_wallet)
        lst_results.append(percent_over_1500_final_wallet)
        lst_results.append(percent_over_2000_final_wallet)
        lst_results.append(percent_over_2500_final_wallet)
        lst_results.append(percent_over_3500_final_wallet)

        max_sharpe_ratio = df['sharpe_ratio'].max()
        min_sharpe_ratio = df['sharpe_ratio'].min()
        mean_sharpe_ratio = df['sharpe_ratio'].mean()
        percent_over_1_sharpe_ratio = self.get_percentage_over_threshold_in_df(df, 'sharpe_ratio', 1.0)
        percent_over_2_sharpe_ratio = self.get_percentage_over_threshold_in_df(df, 'sharpe_ratio', 2.0)
        percent_over_3_sharpe_ratio = self.get_percentage_over_threshold_in_df(df, 'sharpe_ratio', 3.0)
        lst_results.append(max_sharpe_ratio)
        lst_results.append(min_sharpe_ratio)
        lst_results.append(mean_sharpe_ratio)
        lst_results.append(percent_over_1_sharpe_ratio)
        lst_results.append(percent_over_2_sharpe_ratio)
        lst_results.append(percent_over_3_sharpe_ratio)

        max_vs_hold_pct = df['vs_hold_pct'].max()
        min_vs_hold_pct = df['vs_hold_pct'].min()
        mean_vs_hold_pct = df['vs_hold_pct'].mean()
        lst_results.append(max_vs_hold_pct)
        lst_results.append(min_vs_hold_pct)
        lst_results.append(mean_vs_hold_pct)

        max_global_win_rate = df['global_win_rate'].max()
        min_global_win_rate = df['global_win_rate'].min()
        mean_global_win_rate = df['global_win_rate'].mean()
        lst_results.append(max_global_win_rate)
        lst_results.append(min_global_win_rate)
        lst_results.append(mean_global_win_rate)

        quantile_10 = df['final_wallet'].quantile(0.9)
        quantile_20 = df['final_wallet'].quantile(0.8)
        quantile_30 = df['final_wallet'].quantile(0.7)
        quantile_40 = df['final_wallet'].quantile(0.6)
        quantile_50 = df['final_wallet'].quantile(0.5)
        quantile_60 = df['final_wallet'].quantile(0.4)
        quantile_70 = df['final_wallet'].quantile(0.3)
        quantile_80 = df['final_wallet'].quantile(0.2)
        quantile_90 = df['final_wallet'].quantile(0.1)
        lst_results.append(quantile_10)
        lst_results.append(quantile_20)
        lst_results.append(quantile_30)
        lst_results.append(quantile_40)
        lst_results.append(quantile_50)
        lst_results.append(quantile_60)
        lst_results.append(quantile_70)
        lst_results.append(quantile_80)
        lst_results.append(quantile_90)

        self.df_benchmark_strategy.loc[len(self.df_benchmark_strategy)] = lst_results

    def benchmark_pair_analyse_values(self, start_date, strategy, df):
        df_pair_backup = df.copy()
        for pair in self.lst_pair:
            df_pair = df_pair_backup.copy()
            df_pair.drop(df_pair[df_pair['pair'] != pair].index, inplace=True)
            lst_results = [start_date, strategy, pair]
            max_final_wallet = df_pair['final_wallet'].max()
            min_final_wallet = df_pair['final_wallet'].min()
            mean_final_wallet = df_pair['final_wallet'].mean()
            percent_over_1000_final_wallet = self.get_percentage_over_threshold_in_df(df_pair, 'final_wallet', 1000)
            percent_over_1500_final_wallet = self.get_percentage_over_threshold_in_df(df_pair, 'final_wallet', 1500)
            percent_over_2000_final_wallet = self.get_percentage_over_threshold_in_df(df_pair, 'final_wallet', 2000)
            percent_over_2500_final_wallet = self.get_percentage_over_threshold_in_df(df_pair, 'final_wallet', 2500)
            percent_over_3500_final_wallet = self.get_percentage_over_threshold_in_df(df_pair, 'final_wallet', 3000)
            lst_results.append(max_final_wallet)
            lst_results.append(min_final_wallet)
            lst_results.append(mean_final_wallet)
            lst_results.append(percent_over_1000_final_wallet)
            lst_results.append(percent_over_1500_final_wallet)
            lst_results.append(percent_over_2000_final_wallet)
            lst_results.append(percent_over_2500_final_wallet)
            lst_results.append(percent_over_3500_final_wallet)

            max_sharpe_ratio = df_pair['sharpe_ratio'].max()
            min_sharpe_ratio = df_pair['sharpe_ratio'].min()
            mean_sharpe_ratio = df_pair['sharpe_ratio'].mean()
            percent_over_1_sharpe_ratio = self.get_percentage_over_threshold_in_df(df_pair, 'sharpe_ratio', 1.0)
            percent_over_2_sharpe_ratio = self.get_percentage_over_threshold_in_df(df_pair, 'sharpe_ratio', 2.0)
            percent_over_3_sharpe_ratio = self.get_percentage_over_threshold_in_df(df_pair, 'sharpe_ratio', 3.0)
            lst_results.append(max_sharpe_ratio)
            lst_results.append(min_sharpe_ratio)
            lst_results.append(mean_sharpe_ratio)
            lst_results.append(percent_over_1_sharpe_ratio)
            lst_results.append(percent_over_2_sharpe_ratio)
            lst_results.append(percent_over_3_sharpe_ratio)

            max_vs_hold_pct = df_pair['vs_hold_pct'].max()
            min_vs_hold_pct = df_pair['vs_hold_pct'].min()
            mean_vs_hold_pct = df_pair['vs_hold_pct'].mean()
            lst_results.append(max_vs_hold_pct)
            lst_results.append(min_vs_hold_pct)
            lst_results.append(mean_vs_hold_pct)

            max_global_win_rate = df_pair['global_win_rate'].max()
            min_global_win_rate = df_pair['global_win_rate'].min()
            mean_global_win_rate = df_pair['global_win_rate'].mean()
            lst_results.append(max_global_win_rate)
            lst_results.append(min_global_win_rate)
            lst_results.append(mean_global_win_rate)

            quantile_10 = df_pair['final_wallet'].quantile(0.9)
            quantile_20 = df_pair['final_wallet'].quantile(0.8)
            quantile_30 = df_pair['final_wallet'].quantile(0.7)
            quantile_40 = df_pair['final_wallet'].quantile(0.6)
            quantile_50 = df_pair['final_wallet'].quantile(0.5)
            quantile_60 = df_pair['final_wallet'].quantile(0.4)
            quantile_70 = df_pair['final_wallet'].quantile(0.3)
            quantile_80 = df_pair['final_wallet'].quantile(0.2)
            quantile_90 = df_pair['final_wallet'].quantile(0.1)
            lst_results.append(quantile_10)
            lst_results.append(quantile_20)
            lst_results.append(quantile_30)
            lst_results.append(quantile_40)
            lst_results.append(quantile_50)
            lst_results.append(quantile_60)
            lst_results.append(quantile_70)
            lst_results.append(quantile_80)
            lst_results.append(quantile_90)

            self.df_benchmark_strategy.loc[len(self.df_benchmark_strategy)] = lst_results

    def benchmark_parameters_analyse(self, start_date, strategy, df):
        lst_results = [start_date, strategy]
        lst_results_backup = lst_results.copy()
        df_strategy_param = df.copy()

        df_strategy_param_quantile = df_strategy_param.copy()
        df_strategy_param_quantile.drop(df_strategy_param_quantile[df_strategy_param_quantile['final_wallet'] < 1000].index, inplace=True)
        df_strategy_param['quantile_75'] = df_strategy_param_quantile['final_wallet'].quantile(0.75)
        df_strategy_param['quantile_50'] = df_strategy_param_quantile['final_wallet'].quantile(0.5)
        df_strategy_param['quantile_25'] = df_strategy_param_quantile['final_wallet'].quantile(0.25)
        df_strategy_param['quantile_01'] = df_strategy_param_quantile['final_wallet'].quantile(0.01)

        df_strategy_param['score'] = 0

        dct_param = {}
        for parameter in conf.config.lst_paramters:
            lst_param = self.get_lst_from_df_column(df, parameter)
            dct_param[parameter] = lst_param

        # Extract the lists from the dictionary
        lst_of_lst_of_params = list(dct_param.values())

        # Generate all combinations
        # lst_all_combinations = list(product(*lst_of_lst_of_params))
        # Generate all combinations and convert tuples to lists
        lst_all_combinations = [list(combination) for combination in product(*lst_of_lst_of_params)]

        lst_quantile = ['quantile_75', 'quantile_50', 'quantile_25', 'quantile_01']

        for lst in lst_all_combinations:
            df_parameter = df_strategy_param.copy()
            for parameter_id, param in zip(conf.config.lst_paramters, lst):
                df_parameter = df_parameter[df_parameter[parameter_id] == param]
            for quantile in lst_quantile:
                df_parameter['score'] = np.where((df_parameter['final_wallet'] >= df_strategy_param[quantile].mean())
                                                 & (df_parameter['vs_hold_pct'] > 0.0)
                                                 & (df_parameter['sharpe_ratio'] > 1.0)
                                                 & (df_parameter['final_wallet'] > 1000),
                                                 df_parameter['score'] +1,
                                                 df_parameter['score']
                                                 )
            score = df_parameter['score'].sum()
            lst_results.extend([score])
            lst_results.extend(lst)
            # GET BEST PAIRS
            df_parameter.drop(df_parameter[df_parameter['final_wallet'] < 1000].index, inplace=True)
            df_parameter.drop(df_parameter[df_parameter['score'] <= 0].index, inplace=True)
            df_parameter.drop(df_parameter[df_parameter['vs_hold_pct'] <= 0.0].index, inplace=True)
            df_parameter.drop(df_parameter[df_parameter['sharpe_ratio'] < 1.0].index, inplace=True)
            df_parameter = df_parameter.sort_values(by='score', ascending=False)
            lst_pair = df_parameter["pair"].tolist()
            if len(lst_pair) > conf.config.NB_PAIRS_SELECTED:
                lst_pair = lst_pair[:conf.config.NB_PAIRS_SELECTED]
                lst_pair_for_in = lst_pair.copy()
            else:
                lst_pair_empty = [""] * (conf.config.NB_PAIRS_SELECTED - len(lst_pair))
                lst_pair_for_in = lst_pair.copy()
                lst_pair = lst_pair + lst_pair_empty
            lst_results.extend(lst_pair)
            lst_pair_in = [0] * len(self.lst_pair)
            lst_results.extend(lst_pair_in)
            self.df_benchmark_parameters.loc[len(self.df_benchmark_parameters)] = lst_results
            for pair in lst_pair_for_in:
                self.df_benchmark_parameters.at[len(self.df_benchmark_parameters)-1, pair] = 1
            lst_results = lst_results_backup.copy()

    def benchmark_pair_analyse(self, start_date, strategy, df_parameters):
        col_name = str(strategy) + "-" + str(start_date)
        self.df_benchmark_pair_lst_compare[col_name] = 0
        for pair in self.lst_pair:
            self.df_benchmark_pair_lst.loc[len(self.df_benchmark_pair_lst)] = 0
            self.df_benchmark_pair_lst.at[len(self.df_benchmark_pair_lst) - 1, "start_date"] = start_date
            self.df_benchmark_pair_lst.at[len(self.df_benchmark_pair_lst) - 1, "strategy"] = strategy
            self.df_benchmark_pair_lst.at[len(self.df_benchmark_pair_lst) - 1, "pair"] = pair
            self.df_benchmark_pair_lst.at[len(self.df_benchmark_pair_lst) - 1, "score"] = df_parameters[pair].sum()

            index_to_replace = self.df_benchmark_pair_lst_compare.index[self.df_benchmark_pair_lst_compare["pair"] == pair][0]
            self.df_benchmark_pair_lst_compare.at[index_to_replace, col_name] = df_parameters[pair].sum()

    def benchmark_pair_final_lst(self, strategy):
        df = self.df_benchmark_pair_lst_compare.copy()
        self.df_benchmark_pair_lst_compare[strategy] = 0
        self.df_benchmark_pair_lst_compare[strategy + "_cpt_0"] = 0
        self.df_benchmark_pair_lst_compare[strategy + "_cpt_10"] = 10

        # Specify the prefix to filter columns
        prefix_to_keep = strategy + "-"
        # Filter columns based on the specified prefix
        filtered_columns = [col for col in df.columns if col.startswith(prefix_to_keep)]
        filtered_df = df[filtered_columns]

        lst_clns = filtered_df.columns.tolist()
        lst_clns = remove_values_from_lst(lst_clns, ["pair"])

        for cln in lst_clns:
            self.df_benchmark_pair_lst_compare[strategy] = np.where(self.df_benchmark_pair_lst_compare[cln] > 0,
                                                                    self.df_benchmark_pair_lst_compare[cln] + self.df_benchmark_pair_lst_compare[strategy],
                                                                    self.df_benchmark_pair_lst_compare[strategy])
            self.df_benchmark_pair_lst_compare[strategy+"_cpt_0"] = np.where(self.df_benchmark_pair_lst_compare[cln] <= 0,
                                                                             1 + self.df_benchmark_pair_lst_compare[strategy+"_cpt_0"],
                                                                             self.df_benchmark_pair_lst_compare[strategy+"_cpt_0"])
            self.df_benchmark_pair_lst_compare[strategy+"_cpt_10"] = np.where(self.df_benchmark_pair_lst_compare[cln] <= 10,
                                                                              1 + self.df_benchmark_pair_lst_compare[strategy+"_cpt_10"],
                                                                              self.df_benchmark_pair_lst_compare[strategy+"_cpt_10"])

    # Function to count values greater than a given threshold in the specified columns
    def count_values_greater_than_threshold(self, row, threshold, columns_to_count):
        return sum(row[col] > threshold for col in columns_to_count)

    def transpose_df(self, df, cln):
        tmp_val = "x"
        df_transposed = df.copy()
        df_transposed.loc[df[cln] < 0, cln] = 0
        # filter
        # df_transposed = df_transposed[df_transposed['vs_hold_pct'] >= 0]
        # df_transposed = df_transposed.drop(df_transposed[df_transposed['vs_hold_pct'] < 0].index)

        # List of columns to keep
        columns_to_keep = conf.config.lst_paramters
        columns_to_keep = ["start", "strategy", "pair"] + columns_to_keep + [cln]

        # Use .loc[] to select only the desired columns
        df_transposed = df_transposed.loc[:, columns_to_keep]

        lst_cln_val = []
        lst_cln_name = []
        for start_date in self.lst_start_date:
            column_name = start_date + "-" + cln
            lst_cln_name.append(column_name)
            df_transposed[column_name] = tmp_val
            df_transposed[column_name] = np.where(df_transposed["start"] == start_date,
                                                  df_transposed[cln],
                                                  df_transposed[column_name])
            lst_val = df_transposed[column_name].tolist()
            lst_val = list(filter(lambda x: x != tmp_val, lst_val))
            lst_cln_val.append(lst_val)

        df_transposed_filtered = df_transposed[df_transposed['start'] == self.lst_start_date[0]]
        df_transposed_filtered = df_transposed_filtered.drop('start', axis=1)
        df_transposed_filtered = df_transposed_filtered.drop(cln, axis=1)

        for start_date, lst_val_for_date in zip(self.lst_start_date, lst_cln_val):
            column_name = start_date + "-" + cln
            df_transposed_filtered[column_name] = lst_val_for_date

        lst_param = df_transposed_filtered.columns.tolist()
        lst_values_to_drop = ['strategy', 'pair'] + lst_cln_name
        self.lst_param_result = [x for x in lst_param if x not in lst_values_to_drop]

        df_transposed_filtered['total_$'] = df_transposed_filtered[lst_cln_name].sum(axis=1)
        df_transposed_filtered['total_$'] = df_transposed_filtered['total_$'] / len(self.lst_start_date)

        # Define the threshold value (e.g., 100)
        threshold_value = 1000
        # Create a new column 'Count_Column' containing the count of values > threshold
        df_transposed_filtered['positive_column'] = df_transposed_filtered.apply(self.count_values_greater_than_threshold,
                                                                                 args=(threshold_value, lst_cln_name),
                                                                                 axis=1)
        threshold_value = 1200
        df_transposed_filtered['profitable_column'] = df_transposed_filtered.apply(self.count_values_greater_than_threshold,
                                                                                   args=(threshold_value, lst_cln_name),
                                                                                   axis=1)

        for start_date in self.lst_start_date:
            column_name = start_date + "-" + cln
            df_transposed_filtered = df_transposed_filtered.drop(df_transposed_filtered[df_transposed_filtered[column_name] <= 0].index)

        df_transposed_filtered['vs_hold_pct_avg'] = 0
        df_transposed_filtered['vs_hold_pct_neg'] = 0
        df_transposed_filtered['sharpe_ratio_avg'] = 0
        df_transposed_filtered['global_win_rate_avg'] = 0
        df_transposed_filtered['global_win_rate_max'] = 0
        df_transposed_filtered['global_win_rate_min'] = 0
        df_transposed_filtered['total_trades'] = 0
        df_transposed_filtered['max_days_drawdown'] = 0

        # df_transposed_filtered = df_transposed_filtered.loc[:100]
        print('len df_transposed_filtered: ', len(df_transposed_filtered))

        if conf.config.MULTI_PROCESS:
            num_cores = multiprocessing.cpu_count()
            # num_processes = os.cpu_count()
            print("cpu count: ", num_cores)

            # Split the DataFrame into chunks based on the number of CPU cores
            chunk_size = len(df_transposed_filtered) // num_cores
            print("chunk_size: ", chunk_size)
            df_chunks = [df_transposed_filtered[i:i + chunk_size] for i in range(0, len(df_transposed_filtered), chunk_size)]

            with multiprocessing.Pool(processes=num_cores) as pool:
                lst_df_results = pool.map(self.multi_thread_transpose_df, df_chunks)

            # Combine the modified rows back into the DataFrame
            df_transposed_filtered = pd.concat(lst_df_results, ignore_index=True)
                
        else:
            df_transposed_filtered = df_transposed_filtered.apply(self.modify_transposed_row, axis=1)

        df_transposed_filtered = df_transposed_filtered[df_transposed_filtered['vs_hold_pct_avg'] >= 0]
        df_transposed_filtered = df_transposed_filtered[df_transposed_filtered['sharpe_ratio_avg'] >= 0]
        df_transposed_filtered = df_transposed_filtered[df_transposed_filtered['total_$'] >= 1000]
        df_transposed_filtered = df_transposed_filtered[df_transposed_filtered['positive_column'] >= int(len(self.lst_start_date)/2)]


        return df_transposed_filtered

    def multi_thread_transpose_df(self, df):
        return df.apply(self.modify_transposed_row, axis=1)

    def modify_transposed_row(self, row):
        strategy = row['strategy']
        pair = row['pair']
        lst_param_row = []
        for param in self.lst_param_result:
            lst_param_row.append(row[param])

        df_fitered = self.df.copy()
        df_fitered = df_fitered[df_fitered['strategy'] == strategy]
        df_fitered = df_fitered[df_fitered['pair'] == pair]

        for param, param_val in zip(self.lst_param_result, lst_param_row):
            df_fitered = df_fitered[df_fitered[param] == param_val]

        row['vs_hold_pct_avg'] = df_fitered['vs_hold_pct'].mean()
        # Count the values below 0 in the specified column
        row['vs_hold_pct_neg'] = (df_fitered['vs_hold_pct'] < 0).sum()
        row['sharpe_ratio_avg'] = df_fitered['sharpe_ratio'].mean()
        row['total_trades'] = df_fitered['total_trades'].sum()
        row['max_days_drawdown'] = df_fitered['max_days_drawdown'].max()
        row['global_win_rate_avg'] = df_fitered['global_win_rate'].mean()
        row['global_win_rate_max'] = df_fitered['global_win_rate'].max()
        row['global_win_rate_min'] = df_fitered['global_win_rate'].min()

        return row