import pandas as pd
from src import utils
from src import feature_eng, data_proc
from src.training import DNNParams, get_train_data, GTrainParams
from src.models.dnn import get_scaler
import numpy as np
import tensorflow as tf
import argparse

root_dir = utils.get_proj_root()
past_data_path = root_dir.joinpath('data/processed/training.csv')
past_data = pd.read_csv(past_data_path, parse_dates=[0], index_col=0)


def get_hr_ft(lookup_dt, feature_name, pred_data, past_data):

    if lookup_dt >= pred_data.index.values[0]:
        try:
            prev_val = pred_data.loc[lookup_dt, feature_name]
            # prev_val = pred_data[pred_data.index==lookup_dt].iloc[0][feature_name]
        except KeyError:
            lookup_dt = lookup_dt + pd.DateOffset(hours=-1)
            # prev_val = pred_data.loc[lookup_dt, feature_name]
            return get_hr_ft(lookup_dt, feature_name, pred_data, past_data)
            # print('b', lookup_dt)

    else:
        try:
            prev_val = past_data.loc[lookup_dt, feature_name]
            # prev_val = past_data[past_data.index==lookup_dt].iloc[0][feature_name]
        except KeyError:
            prev_val = past_data.loc[lookup_dt+pd.DateOffset(hours=-1), feature_name]
            # prev_val = past_data[past_data.index==lookup_dt+pd.DateOffset(hours=-1)].iloc[0][feature_name]

    if isinstance(prev_val, pd.Series):
        prev_val = prev_val[0]

    return prev_val

def get_past_hr_ft(lookup_dt, feature_name,  past_data):
    try:
        prev_val = past_data.loc[lookup_dt, feature_name]
        # prev_val = past_data[past_data.index==lookup_dt].iloc[0][feature_name]
    except KeyError:
        prev_val = past_data.loc[lookup_dt+pd.DateOffset(hours=-1), feature_name]
        # prev_val = past_data[past_data.index==lookup_dt+pd.DateOffset(hours=-1)].iloc[0][feature_name]

    if isinstance(prev_val, pd.Series):
        prev_val = prev_val[0]

    return prev_val


def return_old_temps(dates, return_mean_temp=False) -> pd.DataFrame:

    temp_ft_names = utils.load_value(root_dir.joinpath('feature_store/select_temp_cols.pkl'))
    df = pd.DataFrame({'datetime':dates})

    if return_mean_temp:
        temp_ft_names.append('mean_temp')

    for name in temp_ft_names:
        df[name] = df.datetime.apply(lambda date: get_past_hr_ft(
        lookup_dt=feature_eng.back_n_yr(date, n=1), feature_name=name,
        past_data=past_data
    ))

    return df

def select_temp_cols(temp_input):
    select_temp_ft_names = utils.load_value(root_dir.joinpath('feature_store/select_temp_cols.pkl'))
    select_stns = [int(name[-1]) for name in select_temp_ft_names]
    
    select_stn_idxs = np.array(select_stns) - 1  #index start 0
    stn_idxs = np.arange(0, 28)
    col_to_drop_idxs = list(set(stn_idxs) - set(select_stn_idxs))

    select_temp = temp_input[:, select_stn_idxs]
    mean_temp = temp_input[:, col_to_drop_idxs].mean(axis=1)
    # print(mean_temp.shape, select_temp.shape)
    # select_temp.loc[:, 'mean_temp'] = mean_temp.values
    temp_cols = np.concatenate((select_temp, 
                                mean_temp.reshape(-1, 1)), axis=1)

    temp_cols = pd.DataFrame(data=temp_cols, columns=select_temp_ft_names+['mean_temp'])

    return temp_cols



def init_df(dates, temps=None):
    
    
    temp_ft_names = utils.load_value(root_dir.joinpath('feature_store/select_temp_cols.pkl'))
    

    if temps is None: # get from last year
        df = return_old_temps(dates=dates, return_mean_temp=True)

    elif temps.ndim ==1:
        df =  return_old_temps(dates=dates, return_mean_temp=False)
        df['mean_temp'] = temps
              
    elif temps.ndim==2:
        df = select_temp_cols(temps)
        df['datetime'] = dates

    return df


# input from comp.
future_temp_data = data_proc.get_raw_temp_data(
    root_dir.joinpath('data/raw_data/temp_hist.csv'), training=False)
future_temp_data = future_temp_data[future_temp_data['date'] >='2008']

future_dates = future_temp_data.date + pd.to_timedelta(future_temp_data.hr, unit='h')
future_dates = future_dates.values
future_temps = future_temp_data.iloc[:, 2:].values

def order_inf_data_cols(data:pd.DataFrame):
    pre_inf_cols = utils.load_value(root_dir.joinpath('feature_store/pre_inf_train_ft_col_names.pkl'))
    
    print(pre_inf_cols)
    data = data[pre_inf_cols]
    return data



# another input
def get_input():
    # start_date = '2008-01-01'
    # future_dates = pd.date_range(start=start_date, end='2008-03-01', freq='h', inclusive='left')+ pd.DateOffset(hours=1)

    
    future_temp_data = data_proc.get_raw_temp_data(
        root_dir.joinpath('data/raw_data/temp_hist.csv'), training=False)
    future_temp_data = future_temp_data[future_temp_data['date'] >='2008']

    future_dates = future_temp_data.date + pd.to_timedelta(future_temp_data.hr, unit='h')
    future_dates = future_dates.values
    future_temps = future_temp_data.iloc[:, 2:].values
    print(future_temp_data)
    inf_df = init_df(future_dates, future_temps)
    inf_df = feature_eng.make_featured_data(inf_df, training=False, drop_temp_cols=False)
    inf_df = order_inf_data_cols(inf_df)

    return inf_df

def get_last_n_rows_train_data(n, return_load=True):

    hist_data = get_train_data(GTrainParams.train_data_path)
    hist_data = hist_data.iloc[-n:, :]
    load = hist_data['load'].values
    hist_data.drop(labels='load', inplace=True, axis=1)
    # hist_data = order_inf_data_cols(hist_data)
    if return_load:
        hist_data['load'] = load

    return hist_data

def get_dnn_predictions(inf_df:pd.DataFrame, model):

    dnn_train_params = utils.load_value(fname=root_dir.joinpath('models/train_params_dnn.pkl'))
    dnn_train_params = DNNParams(**dnn_train_params)
    window_size = dnn_train_params.window_size

    init_window_data = get_last_n_rows_train_data(n=window_size-1, return_load=False)
    # inf_time_ft =  list(set(init_window_data.columns) - set(inf_df.columns))

    # create dummy load columns for future df
    # inf_df[inf_time_ft] = np.zeros(shape=(len(inf_df), len(inf_time_ft)))

    
    # merge hist data and current data, extra columns have NA
    inf_df_with_past = pd.concat(objs=(init_window_data, inf_df), axis=0)
    
    # prediction dataframe
    pred_data = pd.DataFrame({'datetime':inf_df.index.values,
                          'load':pd.NA})
    pred_data.set_index('datetime', inplace=True)

    for i in range(len(pred_data)):
        
        curr_datetime = inf_df.index[i]
        prev_1hr = curr_datetime + pd.DateOffset(hours=-1)
        prev_2hr = curr_datetime + pd.DateOffset(hours=-2)
        load_prev_1h = get_hr_ft(prev_1hr, 'load', pred_data=pred_data, past_data=past_data)
        load_prev_2h = get_hr_ft(prev_2hr, 'load', pred_data=pred_data, past_data=past_data)
        
        inf_df_with_past.loc[curr_datetime, 'load_lag1hr'] = load_prev_1h
        inf_df_with_past.loc[curr_datetime, 'load_lag2hr'] = load_prev_2h 

        input_w = inf_df_with_past[i:window_size+i].values
            

        # scale input
        X_scaler, y_scaler = get_scaler()
        input_w = X_scaler.transform(input_w)
        # reshape to batch 1
        input_w = np.reshape(input_w, newshape=(1, input_w.shape[0], input_w.shape[1]))
        pred = model(input_w).numpy().squeeze()

        # rescale output
        pred = y_scaler.inverse_transform(np.array(pred).reshape(-1, 1)).squeeze()
        pred_data.iloc[i] = pred

    return pred_data


def get_gam_predictions(df:pd.DataFrame, model):

    pred_data = pd.DataFrame({'datetime':df.index.values,
                          'load':pd.NA})
    pred_data.set_index('datetime', inplace=True)
    
    for i in range(len(df[:])):

        curr_datetime = df.index[i]

        # inference time features
        prev_1hr = curr_datetime + pd.DateOffset(hours=-1)
        prev_2hr = curr_datetime + pd.DateOffset(hours=-2)
        load_prev_1h = get_hr_ft(prev_1hr, 'load', pred_data=pred_data, past_data=past_data)
        load_prev_2h = get_hr_ft(prev_2hr, 'load', pred_data=pred_data, past_data=past_data)

        features = df.iloc[i, :].values
        features = np.append(features, [load_prev_1h, load_prev_2h])
        features = features.reshape(1, -1)

        pred = model.predict(features)
        pred_data.iloc[i]  = pred[0]

    return pred_data

def get_pred_error(pred_load):
    actual_vals = pd.read_csv(root_dir.joinpath('data/eval_data/Solution.csv'), parse_dates=[0])
    actual_vals['datetime'] = actual_vals.Date + pd.to_timedelta(actual_vals.Hour, unit='h')
    actual_vals.set_index('datetime', inplace=True)

    from sklearn.metrics import mean_absolute_percentage_error

    err= mean_absolute_percentage_error(y_true=actual_vals.Load, y_pred=pred_load)
    return err

def main(model_name):
    inf_df = get_input()
    if model_name == 'linear_gam':
        model = utils.load_model(root_dir.joinpath('models/linear-gam-model.pkl'))
        pred_data = get_gam_predictions(inf_df, model=model)
    elif model_name == 'dnn':
        model = utils.load_model(root_dir.joinpath('models/dnn'))
        pred_data = get_dnn_predictions(inf_df, model=model)

    print(pred_data)
    print(get_pred_error(pred_data.load))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference parser')
    parser.add_argument('--model_name', '-m', type=str, required=True, help='specify model type')
    args = parser.parse_args()
    main(args.model_name)





