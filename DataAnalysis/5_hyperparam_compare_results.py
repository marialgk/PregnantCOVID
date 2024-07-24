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

FILE = 'performance_metrics_risk_20240704_183210.tsv'
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

#%% BOXPLOT

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
boxplot_sorted(df_boxplot, 'accuracy')
boxplot_sorted(df_boxplot, 'balanced_accuracy')
boxplot_sorted(df_boxplot, 'recall')
boxplot_sorted(df_boxplot, 'f1score')
boxplot_sorted(df_boxplot, 'precision')
boxplot_sorted(df_boxplot, 'Jaccard')


#%% PRINT TOP HYPERPARAMS

# Check the best hyperparameter combinations for each performance metrics, for each classifier.

def best_hyper(df1, classifier, score, verbose= True):
    """
    Get the performance scores for a given classifier and a given metric,
    and orders them in descending order. Verbose option that tells how many
    combinations were winners during the loops.

    df = df_boxplot, just like prepared above.
         I should've written a function for df_boxplot, may be an update later.

    Classifier = String. 'RF', 'LGBM', 'XGBoost', 'CatBoost',
                            'SVC', 'GNB', 'KNN, 'LR'

        Score= String. 'accuracy', 'balanced_accuracy', 'precision',
                       'recall', 'f1score', 'ROC_AUC', 'Jaccard'
        '"""
    df1 = df1[(df1['classifier'] == classifier) & (df1['test'] == score)]
    df1 = df1.sort_values(by=['score'], ascending=False)
    df1 = df1[['hyperparams', 'score']]
    if verbose == True:
        unique = len(pd.unique(df1['hyperparams']))
        print(f'There are {unique} hyperparameters combinations.')
    return df1


def best_table(df1, classifier, save=True):
    """
    For a given classifier, a table with the 5 besthyperparameter combinations
    for each of the seven performance metrics (accuracy, balanced accuracyn
    precision, recall, f1-score, ROC-AUC and Jaccard index). I want to check
    which combinations perform the best across the scores, and whether there
    is a lot of variation.
    """

    output = {}
    for i in ['accuracy', 'balanced_accuracy', 'precision',
                'recall', 'f1score', 'ROC_AUC', 'Jaccard']:
        verb = True if i == 'accuracy' else False
        top5 = best_hyper(df1, classifier, i, verbose= verb)[0 : 5]
        top5 = list(top5['hyperparams'])
        output[i] = top5
    out = pd.DataFrame.from_dict(output, orient='index',
                                 columns=['1st', '2nd', '3rd', '4th', '5th'])
    if save == True:
        out.to_csv(f'top5_hyperparams_{classifier}.txt', sep='\t')
    return out

best_table(df_boxplot, 'LR', save=True)


#%% GET PERFOMANCE OF SELECTED HYPERPARAMS

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

sns.boxplot(chosen,
            x="test",
            y="score")



