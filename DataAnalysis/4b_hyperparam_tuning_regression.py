#!/usr/bin/env python
"""
Created on Wed Apr 24 11:59:14 2024

@author: maria
"""

#%% IMPORT LIBRARIES

# Standard libraries
from datetime import datetime

# Matrices and dataframes
import pandas as pd

# Splitting, shuffling, cross validating
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV

# Classifiers
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

# Performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score


#%% LOAD DATASET

FILE = 'weight_unscaled.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='index')

#%% SPLIT TRAIN AND TEST

def separate_X_and_y(df,
                     class_col):
    """
    Gets a dataframe and separates the X and y.
   
      X = dataframe of features.
      y = series of classes.
      """
    X = df.drop(columns=class_col)
    y = df[class_col].to_frame()

    return X, y

def split_dataset(X,
                  y,
                  n_splits=100,
                  test_size=0.2):
    """
    Splits the data into train and test.
    Repeats the operation n times.
    It is stratified, so mantains the class proportion across the splits.

    Input: X as dataframe of features. Y as series or array of classes.
    n_splits is the number of repeats. Test_size is the proportion of the test.

    Output: X train dataframe, X test dataframe, y train series, y test series.
    This function is a generator.
    """
    sss = ShuffleSplit(n_splits=n_splits,
                       test_size=test_size)

    for train_index, test_index in sss.split(X,y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        yield (X_train, X_test, y_train, y_test)


#%% HYPERPARAMETER TUNING

def cv_kfold(X, y,
             n_splits=5,
             n_repeats=25):

    cv = RepeatedKFold(n_splits=n_splits,
                       n_repeats=n_repeats,
                       random_state=42)
    splits = []

    for i in cv.split(X, y):
        splits.append(i)

    return splits


def find_best_hyperparams(clf_model,
                          X,
                          y,
                          param_grid,
                          n_splits=5,
                          n_repeats=25):
    """
    Performs a randomized search, with leave-one-out cross-validation.
    clf_model: model
    param_grid: dictionary with parameters to be tested.

    Outputs dictionary of parameters.
    """

    cv = cv_kfold(X,
                  y,
                  n_splits=n_splits,
                  n_repeats=n_repeats)

    # Grid search optimal parameters
    clf_grid = RandomizedSearchCV(clf_model,
                                  param_grid,
                                  n_iter=100,
                                  cv=cv,
                                  n_jobs=-1,
                                  verbose=2)

    # training model
    clf_grid.fit(X, y.values.ravel())
    return clf_grid.best_params_


def performance(ytest,y_pred):
    """
    Returns list of four performance metrics.
    """
    mse =       mean_squared_error(ytest, y_pred)
    evs = explained_variance_score(ytest, y_pred)
    mxe =                max_error(ytest, y_pred)
    r2s =                 r2_score(ytest, y_pred)

    perf_list = [mse, evs, mxe, r2s]

    return perf_list


def compare_performance(df,
                        models,
                        class_col='group',
                        n_splits=50,
                        test_size=0.2,
                        cv_splits=5,
                        cv_repeats=20):
    """
    This is the collection of the functions above. Like a "main" function.

    INPUT
    df:        Pandas dataframe with one column correspnding to class and the
               others to features.

    class_col  The name of the class column in df.

    models     Nested dictionary. The key is the function call for the model.
               The value is a dictionary of parameters for the grid search.

    n_splits   Number of train_test_splits loops.

    test_size  The proportion of the test samples in the train_test_split at the
               beggining of each loop.

    """
    # Get the dataset and separe features and classes.
    X_full, y_full = separate_X_and_y(df, class_col)

    # Make generator of train-test splits.
    tt_splits = split_dataset(X_full,
                              y_full,
                              n_splits=n_splits,
                              test_size=test_size)

    # Set up empty dictionary
    performance_metrics = {}
    i=0

    # For each split
    for X_train, X_test, y_train, y_test in tt_splits:
        i += 1
        print(f'this is loop number {i}')
        # For each model
        for model, params in models.items():
            print(f'now running {str(model)}')
            # Grid search for top hyperparameters.
            best_params = find_best_hyperparams(model(),
                                                X_train,
                                                y_train,
                                                params,
                                                n_splits=cv_splits,
                                                n_repeats=cv_repeats)
            print(f'found best params for {str(model)}')

             # RE-Fit model
            clf = model(**best_params)
            clf.fit(X_train, y_train)
    
            # Predict values
            y_pred = clf.predict(X_test)
    
            perf = performance(y_test, y_pred)
            performance_metrics[f'{str(model)}_{i}'] = [str(model),
                                                        best_params,
                                                        *perf]
            print('written line')

    # make dataframe
    columns=['classifier',
             'hyperparams',
             'mean_squared_error',
             'explained_var_score',
             'max_error',
             'r2_score']

    performance_metrics = pd.DataFrame.from_dict(performance_metrics,
                                                 orient='index',
                                                 columns=columns)

    return performance_metrics

#%% RUN THE ANALYSIS

models = {SVR: [{'kernel': ['linear', 'rbf', 'sigmoid'],
                 'C':[0.1, 1, 10, 100],
                 'epsilon':[0.001, 0.01, 0.1, 1]},
                {'kernel': ['poly'],
                 'degree': [2, 3, 4, 5],
                 'C':[0.1, 1, 10, 100],
                'epsilon':[0.001, 0.01, 0.1, 1]}],

          LinearRegression: {'fit_intercept':[True, False]},

          DummyRegressor: {'strategy': ['mean']},

          RandomForestRegressor:{'max_depth': [None, 2, 3, 4, 5, 10, 20, 30],  # 6912 comb
                                 'min_samples_leaf': [2, 4, 5, 7],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [50, 100, 200, 600, 1000, 1500],
                                 'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                                 'max_features':['sqrt', 'log2'],
                                 'n_jobs':[-1]},

          KNeighborsRegressor:{'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                'weights':['uniform', 'distance'],
                                'algorithm':['ball_tree', 'kd_tree', 'brute'],
                                'n_jobs':[-1]}
          }

performance_metrics = compare_performance(df,
                                          models,
                                          class_col='weight_birth_neonate1',
                                          n_splits=30,
                                          test_size=0.2,
                                          cv_splits=5,
                                          cv_repeats=25)

#%% SAVE RESULTS 

sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
performance_metrics.to_csv(f'performance_metrics_weight_{sufix}.tsv', sep='\t')
