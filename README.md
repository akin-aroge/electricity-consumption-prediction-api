# Electricity Demand Forecasting

An end-to-end pipeline using temperature data for prediction electricity demand.


## Project Description

The repository includes exploration, training, and inference pipelines for generalized additive model (interpretable) and deep neural networks.

MAPE: 11% (ranked 30th) globally in the BigDEAL forecasting competition.

## Interactive Dashboard
The dashboard provides an interactive interface for viewing the result of different trained models as well as exploring the data.

To launch dashboard:

1. clone the repo.
2. go to the project root
3. make sure you have anaconda installed
4. open the anaconda prompt and run the following commands
```
conda env create -f environment.yml
```

 5. Run the following command at project root

```
streamlit run streamlit/app.py
```