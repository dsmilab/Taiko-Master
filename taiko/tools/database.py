from glob import glob
import pandas as pd

__all__ = ['load_record_df',
           'get_all_drummers',
           'transform_hit_type']


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


def transform_hit_type(label):
    if label in [1, 2]:
        return 1
    if label in [3, 4]:
        return 2
    if label in [5]:
        return 3
    if label in [6]:
        return 4
    if label in [7]:
        return 5

    return 0
