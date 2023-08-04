import pickle
import os
import pathlib
import tensorflow as tf


class Pipe:
    def __init__(self, fns: list):
        self.fns = fns

    def join_pipe(self, another_pipe):
        fn_list = another_pipe.fns
        for fn in fn_list:
            self.fns.append(fn)

    def run_input(self, input):

        for fn in self.fns:
            input_state = fn(input)

        return input_state


def save_value(value, fname):
    with open(fname, "wb") as f:
        pickle.dump(value, f)


def load_value(fname):
    with open(fname, "rb") as f:
        value = pickle.load(f)
    return value


def get_proj_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_feature_names():

    path = get_proj_root().joinpath("feature_store/train_ft_col_names.pkl")
    ft_names = load_value(path)
    return ft_names


def load_model(fname):
    try:
        model = tf.keras.models.load_model(fname)
        # print(model.summary())
    except (ValueError, OSError):
        model = load_value(fname)
    return model
