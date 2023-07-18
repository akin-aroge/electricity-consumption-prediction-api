
import pandas as pd
from src import inference as inf
from src import feature_eng, data_proc
import altair as alt
from src import utils
from src.models import gam_model
import matplotlib.pyplot as plt
import seaborn as sns


root_dir = utils.get_proj_root()


def generate_hr_dates_from_days(n_days, start_date='2008-01-01'):
    """generate hourly datetime for `n_days` starting from `start_date` """

    end_date = pd.to_datetime(start_date) + pd.DateOffset(hours=n_days*24)
    dates = pd.date_range(start=start_date, end=end_date, freq='h', inclusive='left')
    return dates

def form_inf_input(dates, temps=None):
    """make inference dataframe from dates and temps inputs."""

    inf_df = inf.init_df(dates=dates, temps=temps)
    inf_df = feature_eng.make_featured_data(inf_df, training=False, drop_temp_cols=False)
    inf_df = inf.order_inf_data_cols(inf_df)

    return inf_df

def get_chart(data, show_true_vals=True):
    # data.date = data.index
    hover = alt.selection_single(
        fields=["datetime"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    if show_true_vals:
        lines = (
            alt.Chart(data, height=500, title="Electricity Demand")
            .mark_line()
            .encode(
                x=alt.X("datetime", title="Date"),
                y=alt.Y("load", title="Load in Kwh"),
                color='source',
                
            )
        )
    else:
        data = data[data['source']=='predicted']
        lines = (
            alt.Chart(data, height=500, title="Electricity Demand")
            .mark_line()
            .encode(
                x=alt.X("datetime", title="Date"),
                y=alt.Y("load", title="Load in Kwh"),
                
            )
        )



    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(datetime)",
            y="load",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearmonthdate(datetime)", title="date"),
                alt.Tooltip("load", title="load"),
            ],
        )
        .add_selection(hover)
    )
      
    return (lines + points + tooltips).interactive()

def get_temp(n_days):
    # start_date = '2008-01-01'
    # future_dates = pd.date_range(start=start_date, end='2008-03-01', freq='h', inclusive='left')+ pd.DateOffset(hours=1)

    
    future_temp_data = data_proc.get_raw_temp_data(
        root_dir.joinpath('data/raw_data/temp_hist.csv'), training=False)
    future_temp_data = future_temp_data[future_temp_data['date'] >='2008']
    future_temps = future_temp_data.iloc[:, 2:].values

    return future_temps[:n_days*24]

def get_true_vals(n_days):
    root_dir = utils.get_proj_root()
    actual_vals = pd.read_csv(root_dir.joinpath('data/eval_data/Solution.csv'), parse_dates=[0])
    n_pts = n_days*24
    actual_vals = actual_vals.iloc[:n_pts]
    actual_vals['datetime'] = actual_vals.Date + pd.to_timedelta(actual_vals.Hour, unit='h')   
    return actual_vals

def get_pred_data(n_days:int, model_name:str):
    dates = generate_hr_dates_from_days(n_days=n_days)
    temps = get_temp(n_days=n_days)
    inf_df = form_inf_input(dates=dates, temps=temps)

    pred_data = inf.run_inf(model_name=model_name, inf_df=inf_df)
    actual_data = get_true_vals(n_days=n_days)
    plot_data = pd.DataFrame({'datetime':pred_data.index.values, 
                              'predicted': pred_data.load.values,
                              'actual':actual_data.Load.values})
    plot_data = plot_data.melt(id_vars='datetime', var_name='source', value_name='load')
    return plot_data

def plot_gam_partial_dependence(feature_idx, feature_name):
    try:
        model:gam_model.LinGam = utils.load_model(root_dir.joinpath('models/linear-gam-model.pkl')) 
    except FileNotFoundError:
        print("No model file found")

    fig, ax = plt.subplots(figsize=(4,4))
    model_term_idx = feature_idx
    XX = model.generate_X_grid(term=model_term_idx)
    ax.plot(XX[:, model_term_idx], model.partial_dependence(term=model_term_idx, X=XX))
    ax.plot(XX[:, model_term_idx], model.partial_dependence(term=model_term_idx, X=XX, width=.95)[1], c='r', ls='--')

    ax.set_xlabel(feature_name)
    ax.set_ylabel('partial dependence')



    return fig
# def plot_gam_partial_dependence(n_features=4):



#     try:
#         model:gam_model.LinGam = utils.load_model(root_dir.joinpath('models/linear-gam-model.pkl')) 
#     except FileNotFoundError:
#         print("No model file found")
    
#     fig, axs = plt.subplots(1, n_features, figsize=(8*(n_features/2), 5))
#     for i, term in enumerate(model.terms):

#         if term.isintercept:
#             continue
#         try:
#             axs = axs.flatten()
#         except AttributeError:
#             axs = axs
#         ax = axs[i]
#         XX = model.generate_X_grid(term=i)
#         ax.plot(XX[:, i], gam_model.model.partial_dependence(term=i, X=XX), label=str(i))
#         ax.plot(XX[:, i], gam_model.model.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

#     return fig

# def plot_pred_tseries(pred_data:pd.DataFrame):



# start_date = '2008-01-01'
# future_dates = pd.date_range(start=start_date, end='2008-12-31', freq='h', inclusive='left')+ pd.DateOffset(hours=1)

# inf_df = inf.init_df(future_dates)
# inf_df = feature_eng.make_featured_data(inf_df, training=False, drop_temp_cols=False)
# inf_df = inf.order_inf_data_cols(inf_df)
# inf_df.head()

# model = utils.load_value(root_dir.joinpath('models/linear_gam_model.pkl'))

# pred_data= inf.get_predictions(inf_df, model)