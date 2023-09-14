import pandas as pd
from utilities.utils import get_lst_from_df_column
from itertools import product

def filter_df_from_column_values(df, filter):
    lst_strategy = list(set(df["strategy"].tolist()))
    lst_pair = list(set(df["pair"].tolist()))

    # Replace commas with periods in all numeric columns
    df = df.applymap(lambda x: str(x).replace(',', '.'))

    df_filtered = pd.DataFrame(columns=df.columns)

    for strategy in lst_strategy:
        for pair in lst_pair:
            df_tmp = df.copy()
            df_tmp = df_tmp[df_tmp["strategy"] == strategy]
            df_tmp = df_tmp[df_tmp["pair"] == pair]

            lst_cln_to_filter = ["total_$", "positive_column", "sharpe_ratio_avg"]

            df_tmp['total_$'] = df_tmp['total_$'].astype(float)
            df_tmp['positive_column'] = df_tmp['positive_column'].astype(float)
            df_tmp['vs_hold_pct_avg'] = df_tmp['vs_hold_pct_avg'].astype(float)
            df_tmp['sharpe_ratio_avg'] = df_tmp['sharpe_ratio_avg'].astype(float)

            threshold_total = df_tmp["total_$"].quantile(filter)
            threshold_positive_column = df_tmp["positive_column"].quantile(filter)
            threshold_vs_hold_pct_avg = 0
            threshold_sharpe_ratio_avg = 1

            df_tmp = df_tmp[df_tmp["total_$"] >= threshold_total]
            df_tmp = df_tmp[df_tmp["positive_column"] >= threshold_positive_column]
            df_tmp = df_tmp[df_tmp["vs_hold_pct_avg"] >= threshold_vs_hold_pct_avg]
            df_tmp = df_tmp[df_tmp["sharpe_ratio_avg"] >= threshold_sharpe_ratio_avg]

            df_filtered = pd.concat([df_filtered, df_tmp])

    return df_filtered

def filter_and_benchmark_global_results(path, cvs_file):
    df_result = pd.DataFrame()
    df_result_filtered_1 = pd.DataFrame()
    df_result_filtered_2 = pd.DataFrame()

    file_path = path + "/" + cvs_file
    df = pd.read_csv(file_path, sep=";")
    lst_strategy = get_lst_from_df_column(df, 'strategy')
    lst_pair = get_lst_from_df_column(df, 'pair')

    # Replace commas with periods in the 'Column_Name' column
    df['total_$'] = df['total_$'].str.replace(',', '.')
    df['vs_hold_pct_avg'] = df['vs_hold_pct_avg'].str.replace(',', '.')
    df['positive_column'] = df['positive_column'].str.replace(',', '.')

    df['total_$'] = df['total_$'].astype(float)
    df['vs_hold_pct_avg'] = df['vs_hold_pct_avg'].astype(float)
    df['positive_column'] = df['positive_column'].astype(float)

    df['Rank_glb_$'] = df['total_$'].rank(ascending=False).astype(int)
    df['Rank_glb_vs_hold'] = df['vs_hold_pct_avg'].rank(ascending=False).astype(int)
    df['Rank_glb_positive'] = df['positive_column'].rank(ascending=False).astype(int)

    for strategy in lst_strategy:
        # df_filtered = df[(df['strategy'] == strategy) & (df['pair'] == pair)].copy()
        df_filtered = df[(df['strategy'] == strategy)].copy()
        df_filtered['Rank_strg_$'] = df_filtered['total_$'].rank(ascending=False).astype(int)
        df_filtered['Rank_strg_vs_hold'] = df_filtered['vs_hold_pct_avg'].rank(ascending=False).astype(int)
        df_filtered['Rank_strg_positive'] = df_filtered['positive_column'].rank(ascending=False).astype(int)
        df_tmp = df_filtered.copy()
        for pair in lst_pair:
            df_filtered = df_tmp.copy()
            df_filtered_pair = df_filtered[(df_filtered['pair'] == pair)].copy()
            df_filtered_pair['Rank_pair_$'] = df_filtered_pair['total_$'].rank(ascending=False).astype(int)
            df_filtered_pair['Rank_pair_vs_hold'] = df_filtered_pair['vs_hold_pct_avg'].rank(ascending=False).astype(int)
            df_filtered_pair['Rank_pair_positive'] = df_filtered_pair['positive_column'].rank(ascending=False).astype(int)

            df_result = pd.concat([df_filtered_pair, df_result], ignore_index=True)

    df_result["pair_score"] = df_result['Rank_pair_$'] + df_result['Rank_pair_positive']

    for strategy in lst_strategy:
        # df_filtered = df[(df['strategy'] == strategy) & (df['pair'] == pair)].copy()
        df_filtered = df_result[(df_result['strategy'] == strategy)].copy()
        df_tmp = df_filtered.copy()
        for pair in lst_pair:
            df_filtered = df_tmp.copy()
            df_filtered_pair = df_filtered[(df_filtered['pair'] == pair)].copy()
            df_filtered_pair_1 = df_filtered_pair.copy()
            df_filtered_pair_2 = df_filtered_pair.copy()
            column_name = 'pair_score'
            df_filtered_pair_1 = df_filtered_pair_1.nsmallest(1, column_name)
            df_filtered_pair_2 = df_filtered_pair_2.nsmallest(2, column_name)

            df_result_filtered_1 = pd.concat([df_filtered_pair_1, df_result_filtered_1], ignore_index=True)
            df_result_filtered_2 = pd.concat([df_filtered_pair_2, df_result_filtered_2], ignore_index=True)

    return df_result, df_result_filtered_1, df_result_filtered_2

def best_parameters_global_results(path, cvs_file):
    df_result = pd.DataFrame()

    file_path = path + "/" + cvs_file
    df = pd.read_csv(file_path, sep=";")
    lst_strategy = get_lst_from_df_column(df, 'strategy')
    lst_pair = get_lst_from_df_column(df, 'pair')

    start_column = 'pair'
    # Find the first column that contains "INTRV"
    end_column = next(col for col in df.columns if "INTRV" in col)

    # Get the index positions of the start and end columns
    start_index = df.columns.get_loc(start_column)
    end_index = df.columns.get_loc(end_column)

    # Get the list of column names between the start and end columns
    lst_parameters = df.columns[start_index + 1:end_index].tolist()


    for strategy in lst_strategy:
        # df_filtered = df[(df['strategy'] == strategy) & (df['pair'] == pair)].copy()
        df_filtered = df[(df['strategy'] == strategy)].copy()
        df_tmp = df_filtered.copy()
        for pair in lst_pair:
            df_filtered = df_tmp.copy()
            df_filtered_pair = df_filtered[(df_filtered['pair'] == pair)].copy()

            dct_param = {}
            for parameter in lst_parameters:
                lst_param = get_lst_from_df_column(df_filtered_pair, parameter)
                dct_param[parameter] = lst_param

            # Extract the lists from the dictionary
            lst_of_lst_of_params = list(dct_param.values())

            # Generate all combinations
            # lst_all_combinations = list(product(*lst_of_lst_of_params))
            # Generate all combinations and convert tuples to lists
            lst_all_combinations = [list(combination) for combination in product(*lst_of_lst_of_params)]
