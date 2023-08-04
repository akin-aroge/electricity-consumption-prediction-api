from src import data_proc, feature_eng
from src import utils
import numpy as np
from src.models import gam_model
import argparse
from dataclasses import dataclass
import dataclasses
from sklearn.model_selection import train_test_split
import pathlib
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from src.models import dnn
import matplotlib.pyplot as plt
import sys
from loguru import logger


root_dir = utils.get_proj_root()
# max_year = 2007
# TEST_SIZE_FRAC = 0.3
load_data_path = root_dir.joinpath("data/raw_data/load_hist.csv")
temp_data_path = root_dir.joinpath("data/raw_data/temp_hist.csv")


@dataclass
class GTrainParams:
    test_size_frac = 0.3
    max_year = 2007
    train_data_path = root_dir.joinpath("data/processed/training.csv")
    # name: str #= field(default='def', init=False)


@dataclass
class LinGamParams(GTrainParams):
    lams: np.ndarray
    name = "linear_gam"


@dataclass
class DNNParams(GTrainParams):
    lr: float
    batch_size: int = 32
    window_size: int = 32
    n_epochs: int = 50
    optimizer: str = "adam"
    loss: str = "huber"
    early_stop: bool = True
    name = "dnn"


def get_train_data(path: pathlib.Path):

    try:
        train_df = pd.read_csv(path, parse_dates=[0], index_col=0)
    except FileNotFoundError:
        logger.info("creating train data")
        load_temp = data_proc.init_dataset(
            load_dir=load_data_path,
            temp_dir=temp_data_path,
            max_year=GTrainParams.max_year,
        )
        # train_df = feature_eng.make_train_features(load_temp)
        train_df = feature_eng.make_featured_data(
            load_temp, training=True, drop_temp_cols=True
        )
        train_df.to_csv(path)
    return train_df


def _get_xy(input_df, y_label="load"):
    X = input_df.drop(y_label, axis=1).values
    y = input_df[y_label].values

    return X, y


def split_dataset(input_df, test_size_frac):

    X, y = _get_xy(input_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_frac, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    return mape


# another function
def train_model(model, data_path, train_params):
    pass

    def train(model, train_params):
        pass

    pass


def save_model(model, fname: pathlib.Path):
    try:
        model.save(filepath=fname)
    except AttributeError:
        fname.with_suffix(".pkl")
        utils.save_value(model, fname)


def visualize_loss(history, title=None):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = np.arange(len(loss)) + 1
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train(model="linear_gam", data_path=GTrainParams.train_data_path, save=True):

    # TODO: let function take in only model name and other params like save
    # there should be acces to GTrainParams
    # model paramset should seat in model module and imported.

    # TODO: make similar interfaces for model training
    # while letting **keywords handle variations

    train_df = get_train_data(data_path)
    # print('in train', train_df.columns, '\n', train_df.shape)

    X_train, X_test, y_train, y_test = split_dataset(
        train_df, GTrainParams.test_size_frac
    )
    # if isinstance(model, pygam.LinearGAM):
    if model == "linear_gam":

        n_features = X_train.shape[1]
        lams_pace = np.random.uniform(low=1e-3, high=1e3, size=(100, n_features))
        model_params = LinGamParams(lams=lams_pace)
        model = gam_model.LinGam()
        model.train(X=X_train, y=y_train, **dataclasses.asdict(model_params))
        test_error = eval_model(model=model, X_test=X_test, y_test=y_test)

    elif model == "dnn":
        model_params = DNNParams(lr=1e-4)

        model, history = dnn.train_model(
            train_data=(X_train, y_train),
            val_data=(X_test, y_test),
            **dataclasses.asdict(model_params),
        )

        # visualize_loss(history=history)
        test_error = history.history["val_mae"][-1]
    utils.save_value(
        dataclasses.asdict(model_params),
        fname=root_dir.joinpath(f"models/{model_params.name}_train_params.pkl"),
    )
    save_model(model=model, fname=root_dir.joinpath(f"models/{model_params.name}"))

    print(f"test error is: {test_error}")
    # utils.save_value(model, fname=root_dir.joinpath(f'models/{model_params.name}.pkl'))
    # load_model = utils.load_value(root_dir.joinpath(f'models/{model_params.name}.pkl'))
    # print(load_model)
    # return model


def main(model_name):
    train(model_name)
    # print(utils.get_feature_names())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training parser")
    parser.add_argument(
        "--model_name", "-m", type=str, required=True, help="spec. model type"
    )
    args = parser.parse_args()
    main(args.model_name)
