# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:23:02 2024

@author: maria
"""

#%% IMPORT STUFF

# Standard libraries
import os

# Matrices and dataframes
import pandas as pd

#%% LOAD DATASET

os.chdir(PATH)

FILE = 'imputed_dataset_knn3.tsv'
df = pd.read_csv(FILE, sep='\t', header=0, index_col='index')

#%% LOAD FILE THAT HAS THE WEIGHT INFORMATIOB

FILE = 'final_dataset.tsv'
df_full = pd.read_csv(FILE, sep='\t', header=0, index_col='hash')
df_weight = df_full[['weight_birth_neonate1']]

#%% Merge

df_all = pd.merge(df,
                  df_weight[['weight_birth_neonate1']],
                  how='left',
                  left_index=True,
                  right_index=True)

df_all = df_all[df_all['weight_birth_neonate1'].notna()]

df_all.to_csv('weight_unscaled.tsv', sep='\t')
