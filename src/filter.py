import pandas as pd


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
