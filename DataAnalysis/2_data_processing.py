# -*- coding: utf-8 -*-

#%%########################### IMPORT STUFF ###################################

# Matrices and dataframes
import pandas as pd
from pandas.core.groupby.groupby import MultiIndex

# Data pre-processing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


#%%############################ LOAD DATASET ##################################

FILE = 'final_dataset.tsv'
df_full = pd.read_csv(FILE, sep='\t', header=0, index_col='hash')
df_full = df_full.drop(columns='Unnamed: 0')

#%%################### COUNT MISSING DATA IN ALL THE DATASET ###################


def count_all_missing(df):
    '''
    Returns missing data percentages and values.
    Used for controlling how much data I am imputing or cutting out.

    Input: pandas dataset (patients vs. features)
    '''
    missing = df.isna().sum().sum()
    cell_count = df.shape[0] * df.shape[1]
    perc_missing = 100 * (missing / cell_count)
    print(f'There are {df.shape[0]} patients and {df.shape[1]} variables.')
    print(f'There are {cell_count} cells, {missing} of which are nan.')
    print(f'{perc_missing}% is missing.')

count_all_missing(df_full)


#%%######################### SELECT ROWS AND COLUMNS ##########################

# Select only patients that were assigned to a group
df = df_full[df_full['group'].notna()]

columns = ['weight_gain_during_preg',
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
           'works',
           'group',
           'race_marginalized_group',
           'prenatal_appointments_sum',
           'violent_crimes']

df = df[columns]

#%%############################# RECODE GROUP ################################

# The classifier does not understand booleans.
# Change them to numbers.
df = df.replace({False: 0, True: 1})

# The classifier does not understand strings.
# Change the group strings to numbers.
df['group'] = df['group'].replace({'CONTROL': 0, 'COVID': 1})


#%%######################## DATA CLEANING FUNCTIONS ###########################

def clean_column_names(df):
    '''
    Clean invalid characters from dataframe column names.

    Parameters
        df: dataframe with all column sections

    Returns:
        df with cleaned columns.
    '''

    dict_of_str = {
        '<=': 'lte ',
        '>=': 'gte ',
        '<': 'lt ',
        '>': 'gt ',
        '=': 'eq ',
        ',': ' |'}

    nlevels = df.columns.nlevels
    if df.columns is MultiIndex:
        for i in range(nlevels):
            for key,value in dict_of_str.items():
                new_cols = df.columns.levels[i].str.replace(key, value)
                df.columns = df.columns.set_levels(new_cols, level=i)
    else:
        for key,value in dict_of_str.items():
            df.columns = df.columns.str.replace(key, value)

    return df


def count_na(df, axis=1):
    """
    Returns df with count and percentage of missing data for each row/column.
    """
    neg_axis = 1 - axis
    count   = df.isna().sum (axis=neg_axis)
    percent = df.isna().mean(axis=neg_axis) * 100
    df_na   = pd.DataFrame({'percent':percent, 'count':count})
    index_name = 'column' if axis==1 else 'index'
    df_na.index.name = index_name
    df_na.sort_values(by='count', ascending=False, inplace=True)

    return df_na


def normalize_columns_dtypes(df):
    """
    If the data type is set as object, convert into a numeric type.
    """
    grouping = df.columns.to_series().groupby(df.dtypes).groups.items()
    grouping_filtered = dict(filter(lambda item: item[0]=="object", grouping))

    column_to_adjusts=[]
    for _, values in grouping_filtered.items():
        column_to_adjusts = column_to_adjusts + list(values)

    for col in column_to_adjusts:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def drop_na(df, axis=0, percent=30.0, show_dropped=True):
    """
    Removes columns or rows that have too many missing values.
    axis: 1 for columns and 0 for rows.
    percent: threshold of missing values sufficient to drop a column.
    show_dropped: show which columns/rows were dropped.
    """
    df_na = count_na(df, axis=axis)
    to_drop = df_na[df_na['percent'] > percent].index
    if show_dropped:
        print(to_drop)

    df_dropped_na = df.drop(labels=to_drop, axis=axis)

    return df_dropped_na


def variance_threshold(df,threshold):
    """
    Drops columns whose variance is below a certain threshold.
    i.e., drops columns with low variability, as they carry little information.
    """
    var_thres=VarianceThreshold(threshold=threshold)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]

#%%########################### DATA CLEANING ##################################

# Remove invalid characters from column names.
df = clean_column_names(df)

#Convert object dtypes to numeric dtypes.
df = normalize_columns_dtypes(df)

count_all_missing(df)

# Remove features and patients with too many missing values.
# >30% features
df_clean = drop_na(df, axis=1, percent=30, show_dropped=True)

print('Drop columns:')
print(f'Dropped: {set(df.columns) - set(df_clean.columns)}')
print(f'{len(df_clean.columns)} Remaining: {df_clean.columns}')

count_all_missing(df_clean)

# >10% patient
df_clean = drop_na(df_clean, axis=0, percent=20, show_dropped=True)
count_all_missing(df_clean)

# Remove variables with low variability
df_clean_var = variance_threshold(df_clean, 0.05)
count_all_missing(df_clean_var)


#%%############################# NORMALIZE DATA ###############################

scaler = StandardScaler()

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

df_clean_var[non_bin_cols] = scaler.fit_transform(df_clean_var[non_bin_cols])

df_clean_var.to_csv('not_imputed_norm_dataset_7vars.tsv', sep='\t')


#%%############################# KNN IMPUTATION ###############################

# This is the main parameter for the KNN imputation.
# I've run several times with different values.
k_knn = 3

imputer = KNNImputer(n_neighbors=k_knn)
imputed_df = imputer.fit_transform(df_clean_var)
imputed_df = pd.DataFrame(imputed_df,
                          columns=df_clean_var.columns,
                          index=df_clean_var.index)

imputed_df.to_csv(f'imputed_norm_dataset_knn{k_knn}_7vars.tsv', sep='\t')
