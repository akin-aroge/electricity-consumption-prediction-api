"""Data Exploration Page"""

import pandas as pd
import numpy as np
import streamlit as st
from src import utils, data_proc

from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


root_dir = utils.get_proj_root()

load_data_path = root_dir.joinpath('data/raw_data/load_hist.csv')
temp_data_path = root_dir.joinpath('data/raw_data/temp_hist.csv')
max_year = 2007
min_year = 2005
# get data
@st.cache_data
def get_data():
    load_temp_df = data_proc.init_dataset(load_dir=load_data_path,
                                          temp_dir=temp_data_path,
                                          max_year=max_year)
    return load_temp_df


def main():

    st.write(
        """ 
            # Data Exploration
            This section shows an exploratory walkthrough of the input timeseries dataset 
            to the model. This dataset consists of hourly load and temperature 
            values for a region in North Carolina.
"""
    )
    year_filter = year_slider()
    data = filter_data_year(year_filter=year_filter)
    sec_overall_load(data=data)
    sec_temp_timeseries(data=data, temp_stns_filter=True)
    sec_demand_by_time()
    sec_demand_by_month(data=data)
    sec_demand_by_wkday(data=data)
    sec_demand_by_hr(data=data)
    sec_correlations()
    sec_temp_load_corr(data=data)
    sec_temp_lags_corr(data=data)

def year_slider():

    value = st.sidebar.slider(label="Set year interval:", min_value=min_year, 
              max_value=max_year, value=(min_year, max_year))
    return value
    
def filter_data_year(year_filter):
    load_temp = get_data()
    (select_min_year, select_max_year) = year_filter

    
    load_temp = load_temp[(load_temp.datetime < str(select_max_year+1)) & 
                          (load_temp.datetime > str(select_min_year))
                          ]
    return load_temp

def sec_overall_load(data:pd.DataFrame):
    # st.write("""""## Demand Trend Over Time")
    st.write(
        """
## Demand Trend Over Time
The figure shows a plot of hourly electricity demand.
"""
    )

    load_data_raw = data[['load', 'datetime']].set_index('datetime') 
    fig, ax = plt.subplots()
    load_data_raw.load.plot(ax=ax)
    ax.set_ylabel('Load in kWh')
    st_img_show(fig)

    st.write(
        """ 
The demand plot shows an annual periodicity with peaks including a peak around August 
and a smaller peak around February. This months likely represent the hottest and coldest
times of the year where air conditioning and heating needs lead to increase in electricity
demand. In contrastm the troughs around May are likely due to a moderate demand when the 
temperature is closer to room temperature.
"""
    )

def temp_stations():
    temp_stn_range = list(np.arange(1, 10))
    temp_stn_idxs = st.sidebar.multiselect(label="Select temperature stations:",
                                           options=temp_stn_range, 
                                           default=temp_stn_range)
    return temp_stn_idxs

def sec_temp_timeseries( data:pd.DataFrame, temp_stns_filter=False):

    st.write(
        """
        ## Hourly Temperature over time
        The figure shows a plot of hourly temperatures collected across select temperature stations.
        """
    )
    data = data.drop(labels=['load'], axis=1).set_index('datetime')

    if temp_stns_filter:
        temp_stn_idx = temp_stations()
        temp_col_names = ['t'+str(idx) for idx in temp_stn_idx]
        data = data[temp_col_names]

    try:
        fig, ax = plt.subplots()
        data.plot(legend=True, ax=ax)
        ax.set_ylabel("Temperature, $^\circ$F")
        st_img_show(fig)

        st.write(
            """
        The temperatures across the different stations appear to be significantly correlated.
        Here again, we notice an annual pattern which confirms the hypothesis of the electricity demand trend.
        The hottest temperatures occur around July/August (summer), while there are dips in January/February (winter). The peaks around
        August lead to increased air conditioning use and the load plot suggest this may be responsible for 
        highest electricity demand.

        This makes the temperature an important predicitve feature.
        """
        )
    except TypeError:
        st.write("select at least one temperature station to view.")

def sec_demand_by_time():
    st.write(
        """
        ## Demand by Time
        In this section, the distribution across different time levels are examined.
        """

    )

def sec_demand_by_month(data:pd.DataFrame):
    st.write(
            """
            ### Demand by month of the year.
            The plot shows the distribution of the demand by the month of the year.
            """
    )

    temp_df = (data
               .assign(month = data.datetime.dt.month))
    fig, ax = plt.subplots()
    sns.boxplot(x='month', y='load', data=temp_df, color='w', ax=ax)
    set_plot_labels(ax, xlabel="Month")
    st_img_show(fig)
    st.write(
        """
This plot makes clear the distribution of demand for each month showing the median temperature
rises from June to the peak in August followed by a drop in september. The lowest demand
is experienced in April which follows the slight bump in load due to heating deamnd in th winter months
starting from the October dip till February the following year.

It is also noticeable that there is a high variation in the demand when the median is higher, which may
reflect variations in consumer preferences, needs, or ability to meet energy cost.

This makes the month of the year an important predicitve feature.
 
"""
    )

def sec_demand_by_wkday(data:pd.DataFrame):
    st.write(
        """
        ### Demand by day of the week.
        The plot shows the distribution  of the demand for weekday groups.
        """
    )
    temp_df = (data.copy()
           .assign(day_of_wk = data.datetime.dt.day_name())
           )
    fig, ax = plt.subplots()
    sns.boxplot(x='day_of_wk', y='load', data=temp_df, color='w', ax=ax)
    set_plot_labels(ax, xlabel="Weekday")
    st_img_show(fig)
    st.write(
        """
Interestingly, there appears to be no significant differnces in the distribution of loads for 
different days of the week.
"""
    )

def sec_demand_by_hr(data:pd.DataFrame):
    st.write(
        """
### Demand by hour of the day. 
Presumably, elctricity demand is relatively higher at times when people are most likely to be indoors
in the eveninig.
"""
    )
    temp_df = (data.copy()
           .assign(hr = data.datetime.dt.hour)
           )
    fig, ax = plt.subplots()
    sns.boxplot(x='hr', y='load', data=temp_df, color='w', ax=ax)
    set_plot_labels(ax, xlabel="hour")
    st_img_show(fig)

    st.write(
        """
Expectedly, the distribution shows that the electricity demand peaks in the evening hours around 
17:00. Demand appears to be lowest just before dawn at 5:00. It appears that a descent in demand start
from later in the evening up until early mornings.

THis suggests a significant non-linear variability of load with hour which will be considered in the modelling.
"""
    )

def sec_correlations():
    st.write(
        """
## Correlations
We would now look a few correlations.
"""
    )

def sec_temp_load_corr(data):
    st.write(
        """
### Load Vs. Temperature
Let's examine the correlation between temperature and the demand
"""
    )

    temp_df = (data.copy()
           .assign(temp = data.iloc[:, 4])
           .assign(month = data.datetime.dt.month)
           )
    fig, ax = plt.subplots()
    sns.scatterplot(x='temp', y='load', data=temp_df, alpha=0.2, size=0.2, hue='month', ax=ax)
    set_plot_labels(ax, xlabel="temp")
    st_img_show(fig)
    st.write(
        """
The previous plots have clearly suggested that electricity demand increases in the winter and summer months.
This results in the u-sahped plot seen in the plot, showing the non-linearily that exist between
load and temperature.
"""
    )

def sec_temp_lags_corr(data):
    st.write(

        """
        # Temperature Lag correlations
        From one hour to the next, the temperatrure is load demand is expected to drop only slightly which would 
        result in a correlation between lags. 
        """
    )

    temp_df = (data.copy()
           .assign(lag1=data.load.shift(1))
           .assign(lag2=data.load.shift(2))
           .assign(lag7=data.load.shift(7))
           .assign(lag24=data.load.shift(24))
           )
    fig, axs = plt.subplots(2,2)
    sns.scatterplot(x='lag1', y='load', data=temp_df, alpha=0.2, size=0.2, ax=axs[0,0])
    sns.scatterplot(x='lag2', y='load', data=temp_df, alpha=0.2, size=0.2, ax=axs[0,1])
    sns.scatterplot(x='lag7', y='load', data=temp_df, alpha=0.2, size=0.2, ax=axs[1,0])
    sns.scatterplot(x='lag24', y='load', data=temp_df, alpha=0.2, size=0.2, ax=axs[1,1])
    # set_plot_labels(axs, xlabel="temp")
    fig.tight_layout()
    st_img_show(fig)

    st.write(
        """
The plot shows a correlation between the demand and the 1 hr lag. This correlation appears to reduce with
increasing number of hours of the lag but becomes high again after 24 hours, due to same hour trends 
in demand.

Depending on the type of model being used, lag features may be engineered
        in the model building process.
"""
    )


    

def st_img_show(fig:matplotlib.figure.Figure):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)

def set_plot_labels(ax, xlabel='datetime', ylabel='load, kWh'):

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
