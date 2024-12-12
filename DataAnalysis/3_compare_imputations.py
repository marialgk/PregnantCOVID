# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:18:08 2024

@author: maria

This script compares different K values for the KNN imputation and evaluates
its impact on the results.
"""
#%% IMPORT LIBRARIES

# Standard libraries
import os

# Matrices and dataframes
import pandas as pd

# Scikit learn
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score

# Figures and plotting
import matplotlib.pyplot as plt
import seaborn as sns

#%% LOAD DATA

os.chdir(PATH)

# I evaluated 5 K values: 3, 4, 5, 7, and 10. And also no imputation at all.
no_imput  = pd.read_csv('not_imputed_dataset.tsv',   sep='\t', index_col= 'index')
k3_imput  = pd.read_csv('imputed_dataset_knn3.tsv',  sep='\t', index_col= 'index')
k4_imput  = pd.read_csv('imputed_dataset_knn4.tsv',  sep='\t', index_col= 'index')
k5_imput  = pd.read_csv('imputed_dataset_knn5.tsv',  sep='\t', index_col= 'index')
k7_imput  = pd.read_csv('imputed_dataset_knn7.tsv',  sep='\t', index_col= 'index')
k10_imput = pd.read_csv('imputed_dataset_knn10.tsv', sep='\t', index_col= 'index')

datasets = {'No imputation'           : no_imput,
            'KNN imputer with k = 3'  : k3_imput,
            'KNN imputer with k = 4'  : k4_imput,
            'KNN imputer with k = 5'  : k5_imput,
            'KNN imputer with k = 7'  : k7_imput,
            'KNN imputer with k = 10' : k10_imput}

#%% COMPARE IMPUTED AND ORIGINAL DATA

def melt_table(key_value):
    """
    For producing the figures, I need to reformat the dataset into a 3-column
    format: index, variable and value.
    """

    # Keep only the non binary columns.
    # It makes no sense to represent them in the boxplot.
    non_bin_cols = ['weight_gain_during_preg',
                    'dad_age',
                    'EPS10',
                    'income',
                    'lifetime_stress_summary',
                    'mom_age',
                    'mom_education',
                    'pregnancy_stress_summary',
                    'prenatal_bmi',
                    'STAIS',
                    'STAIT',
                    'prenatal_appointments_sum',
                    'violent_crimes']

    non_bin_df = key_value[1][non_bin_cols]

    df_melt = non_bin_df.melt()
    df_melt['origin'] = key_value[0]
    return df_melt

# Joining the tables with different K values for plotting them.
plot_df = pd.concat([melt_table(i) for i in datasets.items()])
plot_df = plot_df.reset_index(drop=True)

# Boxplot for comparing the distribution of the variables among the various
# k values after the knn imputation (or no imputation at all).
fig, ax1 = plt.subplots(figsize=(20, 10))
sns.boxplot(data=plot_df,
             x='variable',
             y='value',
             hue='origin',
             ax=ax1)
ax1.tick_params(axis='x', labelrotation=90)
plt.savefig('fig_compare_knn_.png')


#%% COMPARE IMPUTED AND ORIGINAL DATA

# The following lines are adapted from my main machine learning script.
# It compares the performance metrics from models trained on the different
# imputation methods. For example, I have the non-imputed dataset, the imputed
# with k=3, the imputed with k=5... So I ran all the LGBM for them and compared
# their performances.


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
                  train_index,
                  test_index):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        yield (X_train, X_test, y_train, y_test)


def performance(ytest,y_pred):

    acc =          accuracy_score(ytest, y_pred)
    bac = balanced_accuracy_score(ytest, y_pred)
    pre =         precision_score(ytest, y_pred)
    rec =            recall_score(ytest, y_pred)
    f1s =                f1_score(ytest, y_pred)
    roc =           roc_auc_score(ytest, y_pred)
    jac =           jaccard_score(ytest, y_pred)

    perf_list = [acc, bac, pre, rec, f1s, roc, jac]

    return perf_list


def run_xgb(datasets_list):

    # Split into X and y
    datasets_Xy = {}
    for name, df in datasets_list.items():
        datasets_Xy[name] = separate_X_and_y(df, 'group')

    # Get splits
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
    i = 0

    performance_metrics = {}
    # For each split/shuffle
    for train_index, test_index in sss.split(datasets_Xy[name][0],
                                             datasets_Xy[name][1]):
        i += 1
        print(f'This is split {i}')

        # Split each dataset into train and test
        # There is an additional for loop compared to the main ML script.
        # because now we have several datasets, one for each imputation.
        for key, val in datasets_Xy.items():
            print(f'This is dataset {key}')
            X_train, X_test = val[0].iloc[train_index], val[0].iloc[test_index]
            y_train, y_test = val[1].iloc[train_index], val[1].iloc[test_index]

            # Run and performance
            # I choose the XG  because the scikit-learn ones don't 
            # accept the non-imputed data.
            clf = XGBClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
            perf = performance(y_test, y_pred)
            performance_metrics[f'{key}_{i}'] =[key, i, *perf]

    # make dataframe
    columns=['Dataset',
             'loop',
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

perf_metrics = run_xgb(datasets)
perf_metrics.to_csv('performance_imputation.txt', sep='\t')

# Preparing the table for plotting.
df_boxplot = pd.melt(perf_metrics,
                     id_vars=['Dataset', 'loop'],
                     value_vars=['accuracy', 'balanced_accuracy', 'precision',
                                 'recall',   'f1score', 'ROC_AUC', 'Jaccard'],
                     var_name='test',
                     value_name='score')


def boxplot_sorted(data,
                   score):

    df_score = data[data['test'] == score]

    grouped = df_score.groupby('Dataset')
    df2 = pd.DataFrame({col:vals['score'] for col, vals in grouped})
    meds = df2.median().sort_values()

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
    fig_bp.savefig(f'boxplot_{score}_imputation.png', bbox_inches = "tight")
    plt.cla()

boxplot_sorted(df_boxplot, 'ROC_AUC')
boxplot_sorted(df_boxplot, 'accuracy')
boxplot_sorted(df_boxplot, 'balanced_accuracy')
boxplot_sorted(df_boxplot, 'recall')
boxplot_sorted(df_boxplot, 'f1score')
boxplot_sorted(df_boxplot, 'precision')
boxplot_sorted(df_boxplot, 'Jaccard')
