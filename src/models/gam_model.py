from pygam import LinearGAM, GAM
from src import utils
import numpy as np

import pathlib

root_dir = utils.get_proj_root()
# n_train_features = utils.load_value(root_dir.joinpath('feature_store/train_ft_col_names.pkl'))
# n_train_features.remove('load')

# lams_space = np.random.uniform(low=1e-3, high=1e3, size=(5, n_train_features))




# class LinGamModel():

#     def __init__(self):
#         self.model = LinearGAM()
#         # self.trained_model = None


#     def train(self, X, y,  **kwargs):

#         n_features = X.shape[1]
        
#         progress = kwargs.get('progress', True)
#         lams_space = kwargs.get('lams', None)

#         if lams_space is None:
#             print('none lams')
#             lams_pace = np.random.uniform(low=1e-3, high=1e3, size=(2, n_features))
#         else:
#             print('lams in')

#         # X, y = self._get_xy(input, label_name=label_name)
#         m = self.model
#         m = m.gridsearch(X, y, lam=lams_pace, progress=progress)
#         self.model = m

#         # return m



#     def _get_xy(input_data, label_name):
#         X = input_data.drop(label_name, axis=1).values
#         y = input_data[label_name].values

#         return X, y
    
#     def predict(self, X:np.ndarray):
#         if X.ndim == 1:
#             X.reshape(1, -1)

#         preds = self.model.predict(X)
#         return preds

    # def save_model(self, fname):
    #     model = self.model
    #     utils.save_value(model, fname=fname)

class LinGam(LinearGAM):
    def __init__(self):
        super().__init__()

    def train(self, X, y, **kwargs):
        progress = kwargs.get('progress', True)
        lams_space = kwargs.get('lams', None)
        n_features = X.shape[1]

        if lams_space is None:
            lams_space = np.random.uniform(low=1e-3, high=1e3, size=(2, n_features))

        self.gridsearch(X, y, lam=lams_space, progress=progress)
        
    

