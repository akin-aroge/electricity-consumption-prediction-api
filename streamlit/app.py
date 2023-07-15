import streamlit as st
import numpy as np
import streamlit_utils as st_utils
from src import inference as inf
import pandas as pd
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="TRACE")

st.set_page_config(
    page_title="Electricy Demand Predition",
    layout = "wide"
)

MODEL_CHOICES = {'Gen. Additive Model (GAM)':'linear_gam', 'Deep Neural Network (DNN)':'dnn'}
APP_MODES = ['exploratory analysis', 'inference']

def main():

    st.sidebar.title("What to do")
    st.title("Electricity Demand Forecast")
    st.caption("Scource Code: [link](https://github.com/akin-aroge/electricity-consumption-prediction-api)")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        APP_MODES, index=1)
    if app_mode == 'inference':
        inf_mode()
    
    

def inf_mode():
    ui_model_name = model_select_ui()
    model_name = MODEL_CHOICES[ui_model_name] 

    n_days = pred_horizon()
    dates = st_utils.generate_hr_dates_from_days(n_days=n_days)
    temps = st_utils.get_temp(n_days=n_days)
    inf_df = st_utils.form_inf_input(dates=dates, temps=temps)

    pred_data = inf.run_inf(model_name=model_name, inf_df=inf_df)

    actual_data = st_utils.get_true_vals(n_days=n_days)
    print(actual_data.shape, pred_data.shape)
    print(pred_data.index.shape, pred_data.load.shape, actual_data.Load.shape)
    plot_data = pd.DataFrame({'datetime':pred_data.index.values, 
                              'predicted': pred_data.load.values,
                              'actual':actual_data.Load.values})
    print('here')
    plot_data = plot_data.melt(id_vars='datetime', var_name='source', value_name='load')

    # st.dataframe(pred_data)
    # pred_data = pred_data.reset_index()
    st.markdown(f"## Predicted Load (using {ui_model_name})")
    show_actual_vals = st.checkbox('Show actual load values')
    chart = st_utils.get_chart(plot_data, show_true_vals=show_actual_vals)
    st.altair_chart(chart.interactive(), use_container_width=True)

def model_select_ui():
    st.sidebar.markdown("# Model")
    model_name = st.sidebar.selectbox("choose model: ", MODEL_CHOICES.keys())
    return model_name
    
def pred_horizon():
    st.sidebar.markdown("# Input")
    n_days = st.sidebar.slider('Set number of days (pred. horizon)', min_value=1, max_value=365, value=2, step=1)
    return n_days



if __name__ == "__main__":
    main()