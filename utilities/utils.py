import os

def get_n_columns(df, columns, n=1):
    dt = df.copy()
    for col in columns:
        dt["n"+str(n)+"_"+col] = dt[col].shift(n)
    return dt

def add_n_columns_to_list(lst, name_prefix, n):
    for i in range(n):
        name = f"{name_prefix}{i}"
        lst.append(name)
    return lst

def clean_df_columns(df):
    # Drop columns with names starting with "Unnamed"
    columns_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def move_column_to_first_position(df, column_to_move):
    column = df.pop(column_to_move)
    df.insert(0, column_to_move, column)
    return df

def remove_values_from_lst(lst, lst_values):
    lst = [item for item in lst if item not in lst_values]
    return lst

def remove_duplicates_from_lst(lst):
    return list(dict.fromkeys(lst))

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def get_lst_intervals_name(nb, suffix):
    lst = []
    for i in range(0, nb, 1):
        str_id = str(i) + "_" + suffix
        lst.append(str_id)
    return lst