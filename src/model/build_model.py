from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

CAT_FEATURES = ['home', 'month', 'day_of_week', 'away', 'skies']


class CustomCatBoostRegressor(CatBoostRegressor):

    def fit(self, X, y=None, **fit_params):
        return super().fit(
            X,
            y=y,
            cat_features=CAT_FEATURES,
            **fit_params
        )


def convert_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)