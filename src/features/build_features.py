import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def preprocess(df):
    drop_cols = ['year', 'day']
    promo_mapping = {'YES': 1, 'NO': 0}

    df = df.drop(
        columns=drop_cols
    ).replace({
        'day_night': {'Day': 1, 'Night': 0},
        'cap': promo_mapping,
        'shirt': promo_mapping,
        'fireworks': promo_mapping,
        'bobblehead': promo_mapping
    }).rename(columns={
        'day_night': 'day_game',
        'home_team': 'home',
        'opponent': 'away'
    })

    return df


class BobbleheadTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # assumes X is full data-frame
        X = preprocess(X)
        X = X.drop(columns='attend')
        return X
