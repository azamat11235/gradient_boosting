import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from catboost import CatBoostRegressor

from numpy import ndarray

import warnings
warnings.filterwarnings("ignore")

class Transformer:
    cat_features = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender', 'genres', 'filming_locations', 'directors', 'keywords']
    def fit(self, x, y):
        self.mlb = []
        for cat_feature in Transformer.cat_features:
            if cat_feature == 'keywords':
                mlb = MultiLabelBinarizer(sparse_output=True)
            else:
                mlb = MultiLabelBinarizer()
            feature = x[cat_feature].apply(lambda elem: [elem] if type(elem) is str else elem)
            mlb.fit(feature)
            mlb.classes_ = np.asarray(list(map(lambda x: ' '.join(x.split(', ')), mlb.classes_)), dtype=object)
            self.mlb.append(mlb)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        x = x.copy()
        for i, cat_feature in enumerate(Transformer.cat_features):
            x[cat_feature] = x[cat_feature].apply(lambda elem: [elem] if type(elem) is str else elem)
            if cat_feature == 'keywords':
                x = x.join(pd.DataFrame.sparse.from_spmatrix(self.mlb[i].transform(x.pop(cat_feature)), columns=cat_feature+'_'+self.mlb[i].classes_, index=x.index))
            else:
                x = x.join(pd.DataFrame(self.mlb[i].transform(x.pop(cat_feature)), columns=cat_feature+'_'+self.mlb[i].classes_, index=x.index))
        return x

def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """
    params = {'max_depth':6, 'learning_rate':0.04841199889779091, 'n_estimators':1500}

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)


    y_train = df_train["awards"]
    del df_train["awards"]

    model = Pipeline(steps=[('transformer', Transformer()), ('regressor', CatBoostRegressor(verbose=False, train_dir='/tmp/catboost_info', **params))])

    model.fit(df_train, y_train)
    return model.predict(df_test)
