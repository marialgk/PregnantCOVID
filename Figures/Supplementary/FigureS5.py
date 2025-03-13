# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 01:59:17 2025

@author: maria
"""
#%% IMPORTS

# Standard libraries
from datetime import datetime
import os

# Matrices and dataframes
import pandas as pd

# Figure plotting
import matplotlib.pyplot as plt

# Splitting, shuffling, cross validating
from sklearn.model_selection import StratifiedShuffleSplit

# Classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Performance metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score

#%% Set environment and load data

os.chdir('H:\Meu Drive\parteI\logistic_regression')

FILE = '15vars_complications.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='Unnamed: 0')

df = df.replace({False: 0, True: 1})

df = df.drop(columns='delivery_complications')

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


def performance(ytest,y_pred, y_prob):

    acc =          accuracy_score(ytest, y_pred)
    bac = balanced_accuracy_score(ytest, y_pred)
    pre =         precision_score(ytest, y_pred)
    rec =            recall_score(ytest, y_pred)
    f1s =                f1_score(ytest, y_pred)
    roc =           roc_auc_score(ytest, y_prob)
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
               The value is a dictionary of parameters.

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
        for ident, params_model in models.items():
            print(f'now running {str(ident)}')
            
            # Unwrap params_id
            params = params_model[0]
            model = params_model[1]

             # Fit model
            clf = model(**params)
            clf.fit(X_train, y_train)
    
            # Predict values
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
    
            perf = performance(y_test, y_pred, y_prob)
            performance_metrics[f'{ident}_{i}'] = [str(model),
                                                   params,
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

#%% Run models

models = {'LR': [{'penalty':'l2',
                  'class_weight':'balanced',
                  'solver':'liblinear',
                  'C':1,
                  'n_jobs':-1, 
                  'max_iter':1000, 
                  'verbose':2},
                  LogisticRegression],
          
          'SVM_rbf': [{'C':1000,
                       'gamma':0.01,
                       'kernel':'rbf',
                       'probability': True},
                      SVC],
          
          'SVM_linear':[{'C':10,
                       'gamma':1,
                       'kernel':'linear',
                       'probability': True},
                      SVC],
          
          'SVM_poly':[{'C':0.1,
                       'gamma':1,
                       'kernel':'poly',
                       'probability': True},
                      SVC]}

performance_metrics = compare_performance(df,
                                          models,
                                          class_col='group',
                                          n_splits=30,
                                          test_size=0.2,
                                          cv_splits=5,
                                          cv_repeats=25)

sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
performance_metrics.to_csv(f'performance_kernels_covid_{sufix}.tsv', sep='\t')

#%% PREPARE FOR BOXPLOT

df = performance_metrics.reset_index(drop=False)

def remove_trail(string):
    return '_'.join(string.split('_')[0:-1])

df['index'] = df['index'].apply(remove_trail)

# Prepare table for plotting.
df_boxplot = pd.melt(df,
                     id_vars=['index', 'hyperparams'],
                     value_vars=['accuracy', 'balanced_accuracy', 'precision',
                                 'recall',   'f1score', 'ROC_AUC', 'Jaccard'],
                     var_name='test',
                     value_name='score')

#%% BOXPLOT

def boxplot_sorted(data,
                   score):

    df_score = data[data['test'] == score]

    grouped = df_score.groupby('index')
    df2 = pd.DataFrame({col:vals['score'] for col, vals in grouped})
    meds = df2.median().sort_values()
    df2[meds.index].boxplot(rot=90)

    fig_bp, ax_bp =plt.subplots(figsize=(12,12))

    df2[meds.index].boxplot(rot=90,
                            fontsize=16,
                            boxprops=dict(linewidth=3,
                                          color='cornflowerblue'),
                            whiskerprops=dict(linewidth=3,
                                              color='cornflowerblue'),
                            medianprops=dict(linewidth=3,
                                             color='firebrick'),
                            capprops=dict(linewidth=3,
                                          color='cornflowerblue'),
                            flierprops=dict(marker='o',
                                            markerfacecolor='dimgray',
                                            markersize=8,
                                            markeredgecolor='black'),
                            ax=ax_bp)
    fig_bp.savefig(f'boxplot_{score}_covid_SD_13032025.png')
    plt.cla()


boxplot_sorted(df_boxplot, 'ROC_AUC')
boxplot_sorted(df_boxplot, 'accuracy')
boxplot_sorted(df_boxplot, 'balanced_accuracy')
boxplot_sorted(df_boxplot, 'recall')
boxplot_sorted(df_boxplot, 'f1score')
boxplot_sorted(df_boxplot, 'precision')
boxplot_sorted(df_boxplot, 'Jaccard')