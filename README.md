# Electricity Demand Forecasting

An end-to-end pipeline using temperature data for prediction electricity demand.


## Project Description

The repository includes exploration, training, and inference pipelines for generalized additive model (interpretable) and deep neural networks.

MAPE: 11% (ranked 30th) globally in the BigDEAL [Global Energy forecasting competition](https://en.wikipedia.org/wiki/Global_Energy_Forecasting_Competition).

## Dashboard Snapshot
The [interactive UI](https://electricity-demand-prediction.streamlit.app/) provides an interface for viewing the result of different trained models as well as exploring the data. A snapshot of the dashboard is available here:

![dashboard_snapshot](./reports/ui_snapshot.PNG?raw=true)

## Reproducing Dashboard
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
streamlit run app.py
```

## Project Organization

-------------------------
```
.
├── data
│   ├── eval_data
│   ├── processed
│   └── raw_data
├── feature_store
├── models
│   ├── dnn
│   │   ├── assets
│   │   └── variables
│   └── lstm_model_0
│       ├── assets
│       └── variables
├── notebook
│   └── tutorial
│       └── data
├── reports
├── src
│   └── models
└── streamlit_utils
```

## Technologies

- Numpy
- TensorFlow
- Scipy
- Scikit-learn
- matplotlib
- PyGAM
- etc.