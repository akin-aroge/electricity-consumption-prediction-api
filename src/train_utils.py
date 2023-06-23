

# def get_dnn_predictions(inf_df:pd.DataFrame, model):

#     init_window_data = get_past_n_rows_pre_inf_ft(n=window_size-1, return_load=False)
#     # inf_time_ft =  list(set(init_window_data.columns) - set(inf_df.columns))

#     # create dummy load columns for future df
#     # inf_df[inf_time_ft] = np.zeros(shape=(len(inf_df), len(inf_time_ft)))

    
#     # get inf features
#     inf_df_with_past = pd.concat(objs=(init_window_data, inf_df), axis=0)

#     # input_X = np.concatenate((init_window_data.values, inf_df.values))
    
#     # prediction dataframe
#     pred_data = pd.DataFrame({'datetime':inf_df.index.values,
#                           'load':pd.NA})
#     pred_data.set_index('datetime', inplace=True)

#     for i in range(len(pred_data)):
        
        
#         # if i != 0:
#         # fix inference time features
#         curr_datetime = inf_df.index[i]
#         prev_1hr = curr_datetime + pd.DateOffset(hours=-1)
#         prev_2hr = curr_datetime + pd.DateOffset(hours=-2)
#         load_prev_1h = get_hr_ft(prev_1hr, 'load', pred_data=pred_data, past_data=past_data)
#         load_prev_2h = get_hr_ft(prev_2hr, 'load', pred_data=pred_data, past_data=past_data)
        
#         inf_df_with_past.loc[curr_datetime, 'load_lag1hr'] = load_prev_1h
#         inf_df_with_past.loc[curr_datetime, 'load_lag2hr'] = load_prev_2h 
#         # print(load_prev_1h, load_prev_2h)

#         input_w = inf_df_with_past[i:window_size+i].values

#         # try:
#         #     assert np.isnan(input_w).sum() == 0
#         # except AssertionError:
#         #     print(load_prev_1h, load_prev_2h)
#         #     print(input_w[-1])
            

#         # scale input
#         X_scaler, y_scaler = get_scaler()
#         input_w = X_scaler.transform(input_w)
#         # reshape to batch 1
#         input_w = np.reshape(input_w, newshape=(1, input_w.shape[0], input_w.shape[1]))
#         pred = model(input_w).numpy().squeeze()

#         # rescale output
#         pred = y_scaler.inverse_transform(np.array(pred).reshape(-1, 1)).squeeze()
#         # print('pred', pred)
#         # if np.isnan(pred):
#         #     print(np.isnan(input_w).sum())
#         #     break
#         pred_data.iloc[i] = pred


#     return pred_data