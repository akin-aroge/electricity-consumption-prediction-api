import streamlit as st
import numpy as np
import streamlit_utils as st_utils
from src import inference as inf
import pandas as pd
from loguru import logger
import sys
from io import BytesIO

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

    plot_data = st_utils.get_pred_data(n_days=n_days, model_name=model_name)
    tseries_plot_ui(plot_data=plot_data, ui_model_name=ui_model_name)

    if model_name == "linear_gam":
        gam_partial_dependence_ui()

def model_select_ui():
    st.sidebar.markdown("# Model")
    model_name = st.sidebar.selectbox("choose model: ", MODEL_CHOICES.keys())
    return model_name
    
def pred_horizon():
    st.sidebar.markdown("# Input")
    n_days = st.sidebar.slider('Set number of days (pred. horizon)', min_value=1, max_value=365, value=2, step=1)
    return n_days

def tseries_plot_ui(plot_data, ui_model_name):
    st.markdown(f"## Predicted Load (using {ui_model_name})")
    show_actual_vals = st.checkbox('Show actual load values')
    chart = st_utils.get_chart(plot_data, show_true_vals=show_actual_vals)
    st.altair_chart(chart.interactive(), use_container_width=True)

def gam_partial_dependence_ui():
    st.markdown(f"## Partial dependence plots")
    st.write("The plots show how each selected variable (feature) affects the model's prediction")
    feature_idx_dict = {'Temperature':6, 'hour of day':7, 'month of the year':8}
    ui_select_feature = st.selectbox(label="Select feature to view partial dependence plot:",
                              options=list(feature_idx_dict.keys()))
    feature_idx = feature_idx_dict[ui_select_feature]
    fig = st_utils.plot_gam_partial_dependence(feature_idx=feature_idx, feature_name=ui_select_feature)



    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    # st.pyplot(fig=fig)
    st.image(buf)
    gam_partial_dependence_comments_ui(ui_select_feature)

def gam_partial_dependence_comments_ui(feature_name):
    # TODO: write contents to file and access as dictionary
    
        content = GAM_FEATURE_COMMENTS[feature_name]
        st.markdown(content)

    

GAM_FEATURE_COMMENTS = {'Temperature':f"The result show that the Temperature has a non-linear relationship with the model output. \
        In particular, as temperature increases, the model output (electricity demand) is *marginally* lower up until \
             about ~ 80F which is around room temperature where the demand is at a minimum. Beyond this point \
                the demand increases again. \
                This suggest that heating demands (at low temperatures) results in a surge in electricity demand.\
                    As temperature approaches the room temperature, demand dips and then increases again \
                        for cooling needs beyond  80F.",
                        'hour of day': f"The dependence plot shows the expected non-linear relationship between \
                            the predicted electricity demand for a given hour of day. \
                                Electricity demand is predictably highest in the evenings, with lows in the \
                                    early mornings and much later in the evening.",
                        'month of the year':f"The dependence plot shows that the model captures \
                        a non-linear relationship between electricity demand and the the month \
                        of the year. The pattern is expected considering that \
                        the considered region is North Carolina (NC). The peak around July/August corresponds to the \
                        the month where temperature peaks typically in NC. The minimum points around April \
                        and November correpsonds to times when the temperature are closest to the room temperature. \
                        In January and December, it is generally cold enough to require heating which raises \
                        the electricity demand again."
                        }

if __name__ == "__main__":
    main()