# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:53:20 2024

@author: maria
"""


#%% IMPORT MODULES AND LIBRARIES

# Standard libraries
import os

# Data management
import pandas as pd

# Data plotting
import matplotlib.pyplot as plt
# import matplotlib.colors as mcol
import seaborn as sns


import scipy.cluster.hierarchy as sch

#%% LOAD DATASET

PATH = 'H:/Meu Drive/parteI/machine_learning/v14/clustering/covid_pediatrics_article'
os.chdir(PATH)

data = pd.read_csv('shap_values_LR_test_50X5.txt',
                   header=0,
                   sep='\t',
                   index_col='index')
# data = data.drop(columns=['guessed', 'correct', 'true_freq'])

group = pd.read_csv('imputed_norm_dataset_knn3_15vars.tsv',
                    header=0,
                    sep='\t',
                    index_col='index')
group = group.loc[~group.index.duplicated()]

#%% PROCESSING

# This is the multiple SHAP, ran 125 times.
# As each time 20% of the dataset is test, we've ~24 tested samples per round.
# So 3025 rows in total. 
# For clustering analysis, we will consider the mean.

data_mean = data.groupby(level=0).median()


group['group'] = group['group'].replace({0 : 'Control',
                                         1 : 'COVID-19'})

data_mean = pd.concat([data_mean, group['group']], axis=1)
data_mean = data_mean.sort_values(by='group')

shap_vals = data_mean.drop(columns=['group'])
colours = data_mean[['group']]

#%% HIERARCHICAL CLUSTERING: DENDOGRAM

def hierarch_cluster(data,
                     group,
                     outname='',
                     dist_metrics='correlation',
                     method='average',
                     thresh='default'):

    group = group.replace({'Control':'#d1e0dd', 'COVID-19':'#0B9375'})

    fig = sns.clustermap(data,
                         # row_cluster=False,
                         method=method,
                         metric=dist_metrics,
                         z_score=None,
                          standard_scale=None,
                         figsize=(8, 12),
                        row_colors=group,
                        cmap= 'plasma_r',
                        yticklabels=False)
    plt.title(f'Dendrogram - {method}', fontsize=16)
    plt.ylabel(f'{dist_metrics}', fontsize=16)
    fig.savefig(f'dend_{dist_metrics}_{method}_{outname}_50.png')


hierarch_cluster(shap_vals,
                 colours['group'],
                 outname='_pediatrics',
                 dist_metrics='euclidean',
                 method='ward')

