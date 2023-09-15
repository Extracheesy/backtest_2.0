import os
import shutil
import glob
import pandas as pd

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

def clean_df(df):
    # Identify and drop columns with only one unique element
    # columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    columns_to_drop = [col for col in df.columns if df[col].nunique() == 1 and col != 'strategy' and col != 'pair']

    df = df.drop(columns=columns_to_drop)

    # Convert all columns to string type
    df = df.astype(str)

    # Replace all "." with "," in the entire DataFrame
    # df = df.replace('.', ',', regex=True) # wrong replace all the caratere with "."
    df = df.applymap(lambda x: x.replace('.', ','))

    return df

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

def get_dir_strating_with(path, prefix):
    # Specify the directory path where you want to search for directories
    directory_path = path

    # Use a list comprehension to find all directories starting with "results"
    result_directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d)) and d.startswith(prefix)]

    return result_directories

def copy_files_to_target_dir(directories, target_directory, prefix):
    # Iterate through each directory
    for source_directory in directories:
        # List files in the source directory
        files = os.listdir(source_directory)

        # Iterate through the files and copy those starting with "toto" to the target directory
        for file in files:
            if file.startswith(prefix):
                source_file_path = os.path.join(source_directory, file)
                target_file_path = os.path.join(target_directory, file)
                shutil.copy(source_file_path, target_file_path)


def rm_dir(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its contents have been forcefully deleted.")
    except OSError as e:
        print(f"Error: {e}")

def move_column_first(df, column_to_move):
    # Step 1: Remove the column from the DataFrame
    column_removed = df.pop(column_to_move)

    # Step 2: Reinsert the column at the first position
    df.insert(0, column_to_move, column_removed)

    return df

def merge_csv(csv_dir , prefix, target):
    # Step 2: List CSV files in the directory
    csv_files = glob.glob(csv_dir + "/" + prefix + '*.csv')

    # Check if any CSV files were found
    if not csv_files:
        print("No CSV files starting with " + prefix + " found in the specified directory.")
    else:
        # Step 3 and 4: Read and merge the CSV files into a single DataFrame
        all_data = pd.DataFrame()
        for file in csv_files:
            df = pd.read_csv(file, sep=";", index_col=None)
            suffix = file.split("-")[1]
            suffix = suffix.split(".")[0]
            df["creation_date"] = suffix
            all_data = pd.concat([all_data, df], ignore_index=True)

        # Step 5: Save the merged DataFrame to a new CSV file
        output_csv = csv_dir + "/" + target
        all_data = clean_df_columns(all_data)
        all_data = move_column_first(all_data, "creation_date")
        all_data.to_csv(output_csv, index=False, sep=";")
        print(f"Merged data saved to '{output_csv}'")

def get_lst_from_df_column(df, column):
    return list(dict.fromkeys(df[column].to_list()))

def get_lst_dir_strating_with(directory_path, str_start_with):
    # Use a list comprehension to find all directories starting with "toto"
    lst_directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d)) and d.startswith(str_start_with)]
    return lst_directories
