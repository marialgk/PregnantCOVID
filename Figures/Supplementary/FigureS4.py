# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:01:02 2024

@author: maria
"""

#%%######################## IMPORT LIBRARIES ##################################

# Standard libraries
import os

# Matrices and dataframes
import pandas as pd

# Figure plotting
import matplotlib.pyplot as plt
import seaborn as sns

#%% LOAD DATASET

PATH = 'H:/Meu Drive/parteI/machine_learning/v14/exploratory'
os.chdir(PATH)

FILE = 'imputed_norm_dataset_knn3_15vars.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='index')

df['group'] = df['group'].replace({'control':0, 'covid':1})

#%% Static pairplot

def static(df, drop, hue):
    plt.cla()
    df = df.drop(columns=drop)
    
    f = sns.pairplot(df,
                     height=2.5,
                     hue=hue,
                     palette='pastel',
                     corner=True)
    f.fig.set_size_inches(15,15)
    f.savefig(f'pair_plot_{hue}.png', dpi=600)


drop_cols = ['weight_gain_during_preg',
             'lifetime_stress_summary',
             'mom_education',
             'pregnancy_stress_summary',
             'works',
             'race_marginalized_group']
static(df, drop_cols, 'group')
    