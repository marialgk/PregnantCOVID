# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:00:46 2024

@author: maria
"""

#%%######################## IMPORT LIBRARIES ##################################

# Standard libraries
from datetime import datetime
import os
import statistics

# Matrices and dataframes
import numpy as np
import pandas as pd

# Figure plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

# Perfomance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import shap

#%%############################## LOAD DATASET ################################

PATH = 'H:\Meu Drive\parteI\machine_learning/v14/shap/15vars_pediatrics'
os.chdir(PATH)

FILE = 'imputed_norm_dataset_knn3_15vars.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='index')


#%%################ SHAP CONSOLIDATED ANALYSIS FUNCTIONS ######################

def consolidated_SHAP(df,
                     clf_model,
                     params,
                     group_col,
                     fileprefix,
                     n_splits,
                     n_repeats,
                     calc_perf=False):
    """
        Performs several SHAP analysis (one for each leave-one-out set).
        """

    # Set empty columns of the output matrix
    list_shap_values = []
    list_test_sets   = []
    list_accuracy    = []
    list_f1          = []
    list_roc         = []
    true_guess  = {i:[0, 0] for i in df.index}

    # Set classifier
    clf_model = clf_model(**params)

    # Prepare class and features
    X = df.drop(columns=group_col)
    y = df[group_col].to_frame()
    columns = X.columns
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    counter = 0

    for train_index, test_index in cv.split(X,y):
        counter += 1
        print(f'Running split n°{counter}')

        X_train_shap, X_test_shap = X.iloc[train_index], X.iloc[test_index]
        y_train_shap, y_test_shap = y.iloc[train_index], y.iloc[test_index]

        # training model
        print('Training and testing model...')
        clf_model.fit(X_train_shap, y_train_shap.values.ravel())

        if calc_perf == True:
            y_pred  = clf_model.predict(X_test_shap)

            accuracy = accuracy_score(y_test_shap, y_pred)
            list_accuracy.append(accuracy)

            f1 = f1_score(y_test_shap, y_pred)
            list_f1.append(f1)
            
            roc = roc_auc_score(y_test_shap, y_pred)
            list_roc.append(roc)
            
            y_error = pd.Series(y_pred, name='y_hat',index=y_test_shap.index)
            y_error = pd.concat([y_test_shap, y_error], axis=1)
            correct = (lambda x: 1 if x['group'] == x['y_hat'] else 0)
            y_error['is_correct'] = y_error.apply(correct, axis=1)
            for sample, content in y_error.iterrows():
                true_guess[sample][0] += 1
                if content['is_correct'] == 1:
                    true_guess[sample][1] += 1
            
        # explaining model
        print('Running explainer...')
        explainer   = shap.Explainer(clf_model.predict, X_test_shap)
        shap_values = explainer.shap_values(X_test_shap)

        # for each iteration we save the test_set index and the shap_values
        list_shap_values.append(shap_values)
        list_test_sets.append(test_index)

    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])

    print('Concatanating results...')
    for i in range(1,len(list_test_sets)):
        test_set    = np.concatenate((test_set, list_test_sets[i]),axis=0)
        shap_values = np.concatenate((shap_values,
                                      np.array(list_shap_values[i])),
                                     axis=0)

    # bringing back variable names
    X_test_shap = pd.DataFrame(X.iloc[test_set],columns=columns)

    if calc_perf == True:
        print(f'Mean accuracy: {statistics.mean(list_accuracy)}')
        print(f'Mean f1-score: {statistics.mean(list_f1)}')
        print(f'Mean ROC-AUC : {statistics.mean(list_roc)}')
        
        metrics = pd.DataFrame.from_dict({'Accuracy':list_accuracy,
                                          'F1-score':list_f1,
                                          'ROC-AUC' :list_roc},
                                         orient='columns')
        metrics = metrics.melt()
        
        plt.cla()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.boxplot(data=metrics, x='variable', y='value', ax=ax)
        fig.savefig('metrics_consoliated_shap.png')        
        
        true_guess = pd.DataFrame.from_dict(true_guess,
                                            orient='index',
                                            columns=['guessed', 'correct'])
        true_guess['true_freq'] = true_guess['correct'] / true_guess['guessed']
        return shap_values, X_test_shap, true_guess

    return shap_values, X_test_shap



def violin_plot(shap_values,
                X,
                figname,
                max_vars=10):
    """
    Generates a SHAP violin plot.
    """
    # Generate summary dot plot
    print('Making violin plot...')
    shap.summary_plot(shap_values,
                      X,
                      title="SHAP summary plot",
                      show=False,
                      plot_type='violin',
                      max_display=max_vars)

    # Generate summary bar plot
    plt.savefig(figname, format='pdf', dpi=600, bbox_inches='tight')
    plt.cla()


def main_shap(df,
              clf_model,
              params,
              group_col,
              fileprefix,
              n_splits,
              n_repeats,
              max_vars=10,
              sufix=''):

    print(f'Running experiment: {sufix}')

    shap_values, X_test_shap, true_guess = consolidated_SHAP(df,
                                                             clf_model,
                                                             params,
                                                             group_col,
                                                             fileprefix,
                                                             n_splits,
                                                             n_repeats,
                                                             calc_perf=True)

    shap_values_df = pd.DataFrame(data=shap_values,
                                  columns=X_test_shap.columns,
                                  index=X_test_shap.index)
    shap_values_tg = pd.merge(shap_values_df,
                              true_guess,
                              how='left',
                              left_index=True,
                              right_index=True)
    shap_values_tg.to_csv(f'shap_values_{sufix}_{n_repeats}X{n_splits}.txt',
                         sep='\t')

    # Create file name.
    fileprefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    figname = f'{fileprefix}.pdf'

    # Violin plot
    plt.cla()
    violin_plot(shap_values,
                X_test_shap,
                f'violin_{sufix}_{figname}',
                max_vars=max_vars)

    return shap_values_df, true_guess


#%%#################### RUN CONSOLIDATED SHAP ANALYSIS ########################

fileprefix = f'{PATH}'

a, b = main_shap(df,
          clf_model=LogisticRegression,
          params={'verbose': 0, 'solver': 'liblinear', 'penalty': 'l2', 'n_jobs': -1, 'max_iter': 1000, 'class_weight': 'balanced', 'C': 0.1},
          group_col='group',
          fileprefix=fileprefix,
          n_splits=5,
          n_repeats=30,
          max_vars=30,
          sufix='LR_guesses')