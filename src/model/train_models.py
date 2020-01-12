import os
import time
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump
from src.data import load_bobbleheads_data
from src.features import BobbleheadTransformer
from .build_model import CustomCatBoostRegressor
from .utils import convert_seconds


DATA_DIR = 'data'

# location to save final model
MODEL_DIR = 'models'


def train_model():
    # start timer for training time
    start = time.time()
#=============================================================================
# DATA
#=============================================================================
    print('LOAD DATA')
    print('-' * 30)

    # get data
    dataset = load_bobbleheads_data()

    print(f'Dataset Shape: {dataset.shape}')

    # training data partition
    X_train, X_test, y_train, y_test = train_test_split(
        dataset,
        dataset.attend,
        test_size=0.2,
        random_state=88
    )

    X_train.to_csv(os.path.join(DATA_DIR, 'training', 'train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'training', 'test.csv'), index=False)

    print(f'Train data: {X_train.shape}')
    print(f'Test data:  {X_test.shape}')
    print()

    #=============================================================================
    # MODEL
    #=============================================================================
    print('TRAIN MODEL')
    print('-' * 30)

    # hyper-parameter grid
    param_grid = {
        "estimator__depth": randint(3, 16),
        "estimator__iterations": [x for x in range(50, 2000 + 1, 50)],
        "estimator__learning_rate": [0.001, 0.01, 0.1],
        "estimator__l2_leaf_reg": [1, 3, 5, 7]
    }

    # transformation pipeline
    model_pipeline = Pipeline(steps=[
        ('prep', BobbleheadTransformer()),
        ('estimator', CustomCatBoostRegressor(train_dir=os.path.join(MODEL_DIR, 'catboost_info'),
                                              logging_level="Silent"))
    ])

    # randomized hyper-parameter search
    estimator = RandomizedSearchCV(model_pipeline,
                                   param_grid,
                                   n_iter=1,
                                   cv=5,
                                   scoring='neg_root_mean_squared_error',
                                   random_state=88,
                                   n_jobs=-1)

    # fit models
    estimator.fit(X_train, y_train)

    # export cross-validation results
    results_df = pd.DataFrame.from_dict(estimator.cv_results_)
    results_df.to_csv(os.path.join(MODEL_DIR, 'cross_validation_scores.csv'))

    # best model
    print(f'Best Hyper-parameters:')
    pprint(estimator.best_params_)
    print()

    #=============================================================================
    # EVALUATION
    #=============================================================================
    print('EVALUATE PERFORMANCE')
    print('-' * 30)

    # test set predictions
    y_pred = estimator.predict(X_test)

    # test set evaluation
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # scoring function maximizes a negative value
    print(f'Train RMSE: {-estimator.best_score_:.0f}')
    print(f'Test RMSE: {test_rmse:.0f}')
    print()

    #=============================================================================
    # SAVE MODEL
    #=============================================================================

    # export serialized model
    dump(estimator, os.path.join(MODEL_DIR, 'estimator.joblib'))

    # end timer
    end = time.time()

    # report training time
    total_time = end - start
    print(f'Training time: {convert_seconds(total_time)}')
