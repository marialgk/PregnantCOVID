#!/usr/bin/env python
"""
Created on Wed Apr 24 11:59:14 2024

@author: maria
"""

#%% IMPORT LIBRARIES

# Standard libraries
from datetime import datetime

# Matrices and dataframes
import numpy as np
import pandas as pd

# Splitting, shuffling, cross validating
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Performance metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score


#%% LOAD DATASET

FILE = 'imputed_dataset_knn3.tsv'
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
    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                 test_size=test_size)

    for train_index, test_index in sss.split(X,y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        yield (X_train, X_test, y_train, y_test)


#%% HYPERPARAMETER TUNING

def cv_kfold(X, y,
             n_splits=5,
             n_repeats=25):

    cv = RepeatedStratifiedKFold(n_splits=n_splits,
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
                                  scoring='recall',
                                  n_jobs=-1,
                                  verbose=2)

    # training model
    clf_grid.fit(X, y.values.ravel())
    return clf_grid.best_params_


def performance(ytest, y_pred):
    """
    Returns a list of seven performance metrics.
    """
    acc =          accuracy_score(ytest, y_pred)
    bac = balanced_accuracy_score(ytest, y_pred)
    pre =         precision_score(ytest, y_pred)
    rec =            recall_score(ytest, y_pred)
    f1s =                f1_score(ytest, y_pred)
    roc =           roc_auc_score(ytest, y_pred)
    jac =           jaccard_score(ytest, y_pred)

    perf_list = [acc, bac, pre, rec, f1s, roc, jac]

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
             'accuracy',
             'balanced_accuracy',
             'precision',
             'recall',
             'f1score',
             'ROC_AUC',
             'Jaccard']
    performance_metrics = pd.DataFrame.from_dict(performance_metrics,
                                                 orient='index',
                                                 columns=columns)

    return performance_metrics

#%% RUN THE ANALYSIS

models = {RandomForestClassifier: {'max_depth': [None, 2, 3, 4, 5, 10, 20, 30], 
                                   'min_samples_leaf': [2, 4, 5, 7],
                                   'min_samples_split': [2, 5, 10],
                                   'n_estimators': [50, 100, 200, 600, 1000, 1500],
                                   'criterion': ['gini', 'entropy','log_loss'],
                                   'bootstrap':[True, False],
                                   'max_features':['sqrt', 'log2'],
                                   'n_jobs':[-1]},

          LogisticRegression: [{'penalty':['l2', None],     
                                'class_weight':[None, 'balanced'],
                                'solver':['lbfgs', 'newton-cholesky'],
                                'C':[0.1, 1, 10, 100, 1000],
                                'n_jobs':[-1], 'max_iter':[1000], 'verbose':[2]
                                },

                               {'penalty':['l1', 'l2'],
                                'class_weight':[None, 'balanced'],
                                'solver':['liblinear'],
                                'C':[0.1, 1, 10, 100, 1000],
                                'n_jobs':[-1], 'max_iter':[1000], 'verbose':[2]
                                }],

          LGBMClassifier: [{'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [3],
                            'num_leaves':[8],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[150],
                            'learning_rate':[0.005, 0.01],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [3],
                            'num_leaves':[8],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[100],
                            'learning_rate':[0.025, 0.05],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [3],
                            'num_leaves':[8],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[50],
                            'learning_rate':[0.1, 0.2],
                            'extra_trees':[True]},

                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [4],
                            'num_leaves':[16],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[150],
                            'learning_rate':[0.005, 0.01],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [4],
                            'num_leaves':[16],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[100],
                            'learning_rate':[0.025, 0.05],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [4],
                            'num_leaves':[16],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[50],
                            'learning_rate':[0.1, 0.2],
                            'extra_trees':[True]},
                            
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [5],
                            'num_leaves':[32],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[150],
                            'learning_rate':[0.005, 0.01],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [5],
                            'num_leaves':[32],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[100],
                            'learning_rate':[0.025, 0.05],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [5],
                            'num_leaves':[32],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[50],
                            'learning_rate':[0.1, 0.2],
                            'extra_trees':[True]},
                            
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [6],
                            'num_leaves':[64],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[150],
                            'learning_rate':[0.005, 0.01],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [6],
                            'num_leaves':[64],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[100],
                            'learning_rate':[0.025, 0.05],
                            'extra_trees':[True]},
                            {'objective':['binary'],
                            'boosting_type' : ['dart', 'gbdt'],
                            'max_bin':[50, 150, 255, 300],
                            'max_depth' : [6],
                            'num_leaves':[64],
                            'min_data_in_leaf':[3, 5, 7, 10],
                            'lambda_l1':[1, 0.1, 0.01, 0.0001, 0],
                            'lambda_l2':[1, 0.1, 0.01, 0.0001, 0],
                            'feature_fraction':[0.5, 0.8, 1],
                            'num_iterations':[50],
                            'learning_rate':[0.1, 0.2],
                            'extra_trees':[True]}
                            ],

          XGBClassifier: {'alpha':[1, 0.1, 0.01, 0.0001, 0],
                          'colsample_bytree': [0.6, 0.8, 1.0],
                          'eta':[0.01, 0.05, 0.1, 0.2],
                          'gamma': [0,0.5, 1, 2, 5],
                          'lambda':[1, 0.1, 0.01, 0.0001, 0],
                          'max_depth': [3, 4, 5, 6],
                          'min_child_weight': [3, 6, 10],
                          'subsample': [0.6, 0.8, 1.0]
                           },

          CatBoostClassifier: {'iterations': [50, 100, 150, 300, 500],
                               'depth': [3, 4, 5, 6, 7, 8, 9, 10],
                               'loss_function': ['Logloss', 'CrossEntropy'],
                               'l2_leaf_reg': [1, 0.1, 0.01, 0.0001, 0],
                               'leaf_estimation_iterations': [1, 3, 5],
                               'logging_level':['Silent'],
                               'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                               'min_child_samples':[1, 4, 8, 16, 32]},

          DummyClassifier: {'strategy': ['stratified']},

          SVC:{'C':[0.1, 1, 10, 100, 1000],
               'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
               'kernel':['rbf', 'linear', 'poly']},

          KNeighborsClassifier:{'n_neighbors':[ 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                               11, 12, 13, 14, 15, 16, 17, 18,
                                               19, 20, 21, 22, 23, 24, 25, 26,
                                               27, 28, 29, 30],
                                'weights':['uniform', 'distance'],
                                'algorithm':['ball_tree', 'kd_tree', 'brute'],
                                'n_jobs':[-1]},

          GaussianNB:{'var_smoothing':np.logspace(0,-9, num=100)}
          }

performance_metrics = compare_performance(df,
                                          models,
                                          class_col='group',
                                          n_splits=30,
                                          test_size=0.2,
                                          cv_splits=5,
                                          cv_repeats=25)

sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
performance_metrics.to_csv(f'performance_metrics_classifier_{sufix}.tsv', sep='\t')

