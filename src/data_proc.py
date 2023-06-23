import pandas as pd
import numpy as np
import pickle
from src import utils
import os

# def get_raw_data(dir_path:str, max_year:str) -> tuple:

#     load_data_raw = pd.read_csv(dir_path, parse_dates=[0])
#     temp_data_raw = pd.read_csv(dir_path, parse_dates=[0])

#     temp_data_raw.columns = ['date', 'hr', 'station_id', 'temperature']
#     load_data_raw.columns = ['date', 'hr', 'load']

#     load_data_raw = load_data_raw[load_data_raw.date < max_year+1]
#     temp_data_raw = temp_data_raw[temp_data_raw.date < max_year+1]

#     return (load_data_raw, temp_data_raw)
root_dir = utils.get_proj_root()


def get_raw_load_data(dir_path:str, max_year:int, training=True) -> pd.DataFrame:

    load_data_raw = pd.read_csv(dir_path, parse_dates=[0])
    load_data_raw.columns = ['date', 'hr', 'load']

    load_data_raw = load_data_raw[load_data_raw.date < str(max_year+1)]
    
    col_names = load_data_raw.columns.values

    if training:
        utils.save_value(col_names, root_dir.joinpath('feature_store/raw_load_data_col.pkl'))

    return load_data_raw

def get_raw_temp_data(dir_path:str, max_year=None, training=True) -> pd.DataFrame:

    temp_data_raw = pd.read_csv(dir_path, parse_dates=[0])
    temp_data_raw.columns = ['date', 'hr', 'station_id', 'temperature']

    if max_year is not None:
        temp_data_raw = temp_data_raw[temp_data_raw.date < str(max_year+1)]

    temp_data_raw, temp_cols = _reshape_temp_df(temp_data_raw=temp_data_raw)

    if training:
        # store col names
        utils.save_value(temp_cols, root_dir.joinpath('feature_store/proc_temp_col_names.pkl'))

    return temp_data_raw

def _reshape_temp_df(temp_data_raw:pd.DataFrame):
    temp_data_raw['datetime'] = temp_data_raw.date + pd.to_timedelta(temp_data_raw.hr, unit='h')    
    temp_data_raw['dummy_col_01'] = np.arange(len(temp_data_raw)) % 2

    temp_data_pivot = pd.pivot_table(temp_data_raw, values='temperature', index=['date', 'hr','datetime', 'dummy_col_01'], columns='station_id')
    temp_data_pivot.reset_index(inplace=True)
    # temp_data_pivot.drop(labels=['dummy_col_01', 'date', 'hr'], axis=1, inplace=True)
    temp_data_pivot.drop(labels=['dummy_col_01', 'datetime'], axis=1, inplace=True)

    # rename columns
    str_temp_cols_dict = dict([( col, 't'+str(col)) for col in temp_data_pivot.columns 
                               if isinstance(col, int)])
    temp_cols = list(str_temp_cols_dict.values())
    temp_data_pivot.rename(columns=str_temp_cols_dict, inplace=True)

    return temp_data_pivot, temp_cols

def _add_datatime_col(df):
    
    df['datetime'] = df.date + pd.to_timedelta(df.hr, unit='h')
    return df

def init_dataset(load_dir, temp_dir, max_year:int, training=True) -> pd.DataFrame:

    raw_temp_data = get_raw_temp_data(temp_dir, max_year=max_year, training=training)
    raw_temp_data = (raw_temp_data
                     .pipe(_add_datatime_col)
                     .drop(['date', 'hr'], axis=1))

    # if training:
    raw_load_data = get_raw_load_data(load_dir, max_year, training=training)
    raw_load_data = (raw_load_data
                        # .pipe(_add_datatime_col)
                        .drop(['date', 'hr'], axis=1))
        
    dataset = pd.concat([raw_load_data, raw_temp_data], axis=1)
    # dataset.set_index(keys='datetime', inplace=True)

    return dataset



    








