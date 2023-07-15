
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as hol_calendar
from src import utils, feature_eng
import pandas as pd
from loguru import logger
import sys

    # logger.remove()
    # logger.add(sys.stderr, level="INFO")   
# logger.remove()
logger.add(sys.stderr, level="ERROR") 

root_dir = utils.get_proj_root()

def get_month(df):
    df['month'] = df['datetime'].dt.month
    return df

def get_hr(df):
    df['hr']= df['datetime'].dt.hour
    return df

def back_n_yr(date, n) -> pd.Timestamp:
    new_date = date + pd.DateOffset(years=-n)
    return new_date

def is_holiday(df):
    cal = hol_calendar()
    date_series = df['datetime']
    holidays = cal.holidays(start=date_series.min(), end=date_series.max())
    df['is_holiday'] = np.int16(date_series.isin(holidays))
    return df

def is_weekend(df) -> pd.DataFrame:
    weekend_status = np.uint8(df.datetime.dt.day_of_week > 4)
    df['is_weekend'] = weekend_status
    return df

def temp_lag_1hr(df) -> pd.DataFrame:
    # TODO: should start from exisiting dataset
    df['temp_s1'] = df['mean_temp'].shift(1).fillna(method='backfill')
    return df

def temp_lag_2hr(df) -> pd.DataFrame:
    # TODO: should start from exisiting dataset
    df['temp_s2'] = df['mean_temp'].shift(2).fillna(method='backfill')
    return df

def temp_lag_nhr(df) -> pd.DataFrame:
    # TODO: should start from exisiting dataset
    df['temp_s1'] = df['mean_temp'].shift(1).fillna(method='backfill')
    return df

def load_lag_nhr(df, n) -> pd.DataFrame:
    df['load_shift1hr'] = df['load'].shift(n).fillna(method='backfill')
    return df

def feature_lag_nhr(df, feature, n, fillmethod='backfill') -> pd.DataFrame:
    new_ft_name = feature + '_lag' + str(n) + 'hr'
    df[new_ft_name] = df[feature].shift(n).fillna(method=fillmethod)
    return df

def drop_collinear_cols(df, cols, thresh, training=True) -> pd.DataFrame:

    corr_mat = df[cols].corr().abs()
    corr_mat_u = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
    cols_to_drop = [col for col in corr_mat_u.columns if any(corr_mat_u[col] > thresh)]
    df['mean_temp'] = df[cols_to_drop].mean(axis=1)
    df = df.drop(cols_to_drop, axis=1)

    

    select_cols = [name for name in corr_mat_u.columns if name not in cols_to_drop]
    if training:
        utils.save_value(cols_to_drop, root_dir.joinpath('feature_store/temp_cols_to_drop.pkl'))
        utils.save_value(select_cols, root_dir.joinpath('feature_store/select_temp_cols.pkl'))

    return df #, cols_to_drop



def make_train_features(df:pd.DataFrame, training=True, **kwargs) -> pd.DataFrame:
    # TODO: remove columns and replace calls

    temp_corr_thresh = kwargs.get('thresh', 0.96)
    raw_temp_cols = utils.load_value(root_dir
                                        .joinpath('feature_store/proc_temp_col_names.pkl'))
    # select temp cols
    df = (df
          .pipe(drop_collinear_cols, cols=raw_temp_cols, thresh=temp_corr_thresh))

    df = (df
        #   .pipe(drop_collinear_cols, cols=raw_temp_cols, thresh=raw_temp_cols)
          .pipe(feature_eng.get_hr)
          .pipe(feature_eng.get_month)
          .pipe(feature_eng.is_holiday)
          .pipe(feature_eng.is_weekend)
          .pipe(feature_eng.feature_lag_nhr, feature='mean_temp', n=1)
          .pipe(feature_eng.feature_lag_nhr, feature='mean_temp', n=2)
          .pipe(feature_eng.feature_lag_nhr, feature='load', n=1)
          .pipe(feature_eng.feature_lag_nhr, feature='load', n=2)
          )
    df.set_index(keys='datetime', inplace=True)
    
    col_names = list(df.columns.values)
    if training:
        utils.save_value(col_names, root_dir.joinpath('feature_store/train_ft_col_names.pkl'))
        df.to_csv(path_or_buf=root_dir.joinpath('feature_store/hist_feat_data.csv'))
    return df

b = 'in feature_eng'

def make_featured_data(df:pd.DataFrame, training:bool, drop_temp_cols:bool, **kwargs):
    """ Make feature dataframce for training or inference from initialized input dataframe.
    
    Creates a dataframe with the necessary features for training or inference. Number of
    columns will depend on whether call is made with training set to True of False.

    Args:
        df: input dataframe of date and temperature columns
        training: a boolean value determining whether function is called for training or inference
        drop_temp_cols: a boolean determining whether to trim the number of temperature features
        kwargs: other keyword arguments

    Returns:
        df: returns a dataframe of features
    """

    if drop_temp_cols:
        temp_corr_thresh = kwargs.get('thresh', 0.96)
        raw_temp_cols = utils.load_value(root_dir
                                        .joinpath('feature_store/proc_temp_col_names.pkl'))
        df = (df
            .pipe(drop_collinear_cols, cols=raw_temp_cols, thresh=temp_corr_thresh))       
    
    df = (df
          .pipe(feature_eng.get_hr)
          .pipe(feature_eng.get_month)
          .pipe(feature_eng.is_holiday)
          .pipe(feature_eng.is_weekend)
          .pipe(feature_eng.feature_lag_nhr, feature='mean_temp', n=1)
          .pipe(feature_eng.feature_lag_nhr, feature='mean_temp', n=2)
          )
    df.set_index(keys='datetime', inplace=True)

    col_names_pre_inf = list(df.columns.values)
    

    if training: # additional columns available at training, but not before inference time
        df = (df          
              .pipe(feature_eng.feature_lag_nhr, feature='load', n=1)
              .pipe(feature_eng.feature_lag_nhr, feature='load', n=2))
        train_col_names = list(df.columns.values).remove('load')
        utils.save_value(train_col_names, root_dir.joinpath('feature_store/train_ft_col_names.pkl'))
        df.to_csv(path_or_buf=root_dir.joinpath('feature_store/hist_feat_data.csv'))

        col_names_pre_inf.remove('load')
        utils.save_value(col_names_pre_inf, root_dir.joinpath('feature_store/pre_inf_train_ft_col_names.pkl'))
    
    print(b)   
    logger.remove()
    logger.add(sys.stderr, level="ERROR") 
    logger.info("made feature dataset")

    return df
    
