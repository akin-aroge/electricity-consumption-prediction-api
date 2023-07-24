import streamlit as st
from streamlit_utils import streamlit_utils as st_utils
from loguru import logger
import sys
from io import BytesIO

from streamlit_utils.st_pages import exploration, inference

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
        APP_MODES, index=0)
    if app_mode == 'inference':
        inference.main()
    elif app_mode == 'exploratory analysis':
        exploration.main()
    
if __name__ == "__main__":
    main()