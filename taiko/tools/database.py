from glob import glob
import pandas as pd

__all__ = ['load_record_df',
           'get_all_drummers']


def load_record_df(**kwargs):
    record_files = glob('../data/alpha/*/*/record_table.csv')
    record_dfs = []
    for record_file_path in record_files:
        record_df = pd.read_csv(record_file_path)
        record_dfs.append(record_df)
    record_df = pd.concat(record_dfs, ignore_index=True)

    for key, value in kwargs.items():
        record_df = record_df[record_df[key] == value]

    return record_df


def get_all_drummers():
    record_df = load_record_df()
    drummers = list(record_df['drummer_name'].unique())

    return drummers
