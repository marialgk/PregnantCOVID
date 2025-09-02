# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:21:37 2024

@author: maria
"""
#%% IMPORT MODULES AND LIBRARIES
import os

# Matrices and dataframes
import pandas as pd

# Figure plotting
import matplotlib.pyplot as plt
import seaborn as sns

#%% LOAD DATASET

os.chdir(PATH)

FILE = 'performance_metrics_risk.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='Unnamed: 0')

#%% PREPARE FOR BOXPLOT

# Rename the class names as human readable.
df['classifier'] = df['classifier'].replace({"<class 'sklearn.ensemble._forest.RandomForestClassifier'>" : 'RF',
                                             "<class 'lightgbm.sklearn.LGBMClassifier'>" : 'LGBM',
                                             "<class 'xgboost.sklearn.XGBClassifier'>" : 'XGBoost',
                                             "<class 'catboost.core.CatBoostClassifier'>" : 'CatBoost',
                                             "<class 'sklearn.linear_model._logistic.LogisticRegression'>" : 'LR',
                                             "<class 'sklearn.dummy.DummyClassifier'>" : 'Dummy',
                                             "<class 'sklearn.svm._classes.SVC'>" : 'SVC',
                                             "<class 'sklearn.naive_bayes.GaussianNB'>" : 'GNB',
                                             "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>" : 'KNN'})

# Prepare table for plotting.
df_boxplot = pd.melt(df,
                     id_vars=['classifier', 'hyperparams'],
                     value_vars=['accuracy', 'balanced_accuracy', 'precision',
                                 'recall',   'f1score', 'ROC_AUC', 'Jaccard'],
                     var_name='test',
                     value_name='score')

#%% Figure 2A-C

def boxplot_sorted(data,
                   score):

    df_score = data[data['test'] == score]

    grouped = df_score.groupby('classifier')
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
    fig_bp.savefig(f'boxplot_{score}_covid_SD_052024.png')
    plt.cla()


boxplot_sorted(df_boxplot, 'ROC_AUC')
boxplot_sorted(df_boxplot, 'balanced_accuracy')
boxplot_sorted(df_boxplot, 'recall')


#%% Figure 2D

chosen = df_boxplot[df_boxplot['hyperparams'] == "{'verbose': 2, 'solver': 'liblinear', 'penalty': 'l2', 'n_jobs': -1, 'max_iter': 1000, 'class_weight': 'balanced', 'C': 1}"]

chosen = chosen[chosen['test'].isin(['accuracy', 'f1score', 'ROC_AUC'])]

fig, ax = plt.subplots(figsize=(8, 8))
sns.set_style("whitegrid")
sns.barplot(chosen,
            x="test",
            y="score",
            ax=ax,
            color='cornflowerblue',
            errorbar='sd',
            estimator='mean')



