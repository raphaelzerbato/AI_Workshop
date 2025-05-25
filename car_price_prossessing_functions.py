#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import seaborn as sns
from itertools import product, combinations_with_replacement


# ## Define prossessing functions

# In[85]:


"""
Load data and configs function
"""
# Load  data from file
def load_data(file_path) -> pd.DataFrame:
    """
    Load car data from a CSV, JSON, or Excel file.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        data = pd.read_json(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .json, or .xlsx file.")
    return data


# In[86]:


def process_dates(data) -> pd.DataFrame:
    """
    Process selling dates and extract selling year, selling month and selling week-of-the-year
    """
    df = data.copy()
    # Clean the string (remove the part in parentheses)
    df['saledate'] = df['saledate'].str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()
    # Convert to datetime (handles GMT offsets) - invalid dates will become NaT
    df['saledate'] = pd.to_datetime(df['saledate'], utc=True, errors='coerce')
    # Exclude rows with invalid dates
    nb_invalids = df[df['saledate'].isna()].shape[0]
    if nb_invalids>0:
        print(nb_invalids, 'rows with invalid dates have been dropped')
    df = df[~df['saledate'].isna()]
    #Extract selling year, months and weeks
    df['sellingyear'] = df['saledate'].map(lambda x: x.year)
    df['sellingmonth'] = df['saledate'].map(lambda x: x.month)
    #df['sellingweek'] = df['saledate'].map(lambda x: x.dt.isocalendar().week)
    df['sellingweek'] = df['saledate'].map(lambda x: x.isocalendar().week)
    return df


# In[87]:


def define_market(data, market_vars, market_label='market', market_code='marketid', minsize=20, dropna=True) -> pd.DataFrame:
    """
    Build two new column (marketid, market) for the market IDs
    
    Args:
        - market_vars (list of str): the names of columns in data that define the market
        - minsize (int): minimum market size to allow. Sales in markets with lower sizes will be dropped
    Returns:
        - The dataframe data with two new columns:
            market: a concatenate of values from variables in market_vars
            marketid: numerical code of market

    Notes:
        - The same function can be used to define products
        - process_dates should be run before this function (to keep all products, set minsize=1)
    """    
    df = data.copy()
    df['market'] = df[market_vars[0]].astype('str')
    for var in market_vars[1:]:
        df['market'] = df['market'] + '_' + df[var].astype('str')
    markets_tab = df['market'].value_counts()
    selected_markets = markets_tab.iloc[np.where(markets_tab>=minsize)].index
    nb_droppedmarkets = np.sum(~(markets_tab>=minsize))
    nb_droppedrows = np.sum(~df['market'].isin(selected_markets))
    if dropna:
        df = df[df['market'].isin(selected_markets)]
        print(f"\n{nb_droppedmarkets} markets with sizes lower than {minsize} have been dropped; making {nb_droppedrows} dropped rows \n")
    idmarket_mask = dict([(i, selected_markets[i]) for i in range(len(selected_markets))])
    marketid_mask = dict([(selected_markets[i], i) for i in range(len(selected_markets))])
    df['marketid'] = df['market'].map(marketid_mask)
    return df


# In[88]:


def recode_categories(series, replacement_mask=dict(),  min_frequency=10000) -> pd.Series:
    """
    recode values of a series of strings
    
    Args:
        - series (pd.Series): series of strings in lower cases.
        - replacement_mask (dict): a dictionary that maps some values to be replaced to their new values
        - min_frequency (int): minimum frequency per category to allow
                        all values with a lower frequency will be recoded as 'other'
    Returns:
        - pd.Series
    """
    s = series.copy()
    s = s.replace(replacement_mask)
    frequencies = s.value_counts()
    popular_values = set(frequencies[frequencies>= min_frequency].index)-{''}
    s = s.map(lambda x: x if pd.isna(x) or x in popular_values else 'other')
    return s


# In[89]:


def extract_relevant_variables(data, marketvar='marketid', productvar='make', dropna=True,
                               numerical=[], ordered=[], unordered=[], 
                               ordered_masks=dict(), unordered_masks=dict(), 
                               ordered_min_frequencies=dict(), unordered_min_frequencies=dict()) -> pd.DataFrame:
    """
    Extract and process relevant variables
    
    Args:
        - data (pd.DataFrame): shouldcontaain a 'state' column
        - marketvar (str): the column name in data for market
        - productvar (str): the column name in data for product 
        - numerical, ordered, unordered (list of str): the names of relevant columns that will be set as numerical, ordered categorical and unordered categorical 
        - ordered_masks, unordered_masks (dict of dict): dictionaries of replacement masks to be used as arguments in the recode_categories function for each relevant categorical variable
        - ordered_min_frequencies, unordered_min_frequencies (dict of int): dictionaries that provides the min_frequency argument for the recode_categories function for each relevant categorical variable

    Returns:
        - pd.DataFrame: contains the provided relevant columns along with marketvar and productvar columns

    Notes:
        - The distinction in the type of variables (i.e., numerical, ordered, unordered) can be usefull in a later update that considers missing value imputation using chained equations
        - There is a particular processing for 'color', 'interior' regarding value '—' 
            (which isn't recongnized as a 'minus' symbol in some computers)
    
    """
    df = data.copy()
    if productvar in numerical+ordered+unordered:
        df = df[[marketvar]+['state']+numerical+ordered+unordered]
    else:
        df = df[[marketvar,productvar]+numerical+ordered+unordered]
    for var in ['color', 'interior']:
        if var in df.columns:
            df[var] = df[var].replace({'—':pd.NA})           
    print('\nNumber of missing per relevant variable in the remaining dataframe \n', np.sum(df.isna(), axis=0))
    if dropna:
        print(f"\nA total of {np.sum(np.any(df.isna(), axis=1))} rows with missing data in relevant variables have been dropped \n")
        df = df.dropna()
    for var in ordered:
        df[var] = df[var].astype('string').str.lower()
        df[var] = recode_categories(
            df[var], replacement_mask=ordered_masks[var], min_frequency=ordered_min_frequencies[var]
        )
        df[var] = pd.Categorical(df[var], ordered=True)
    for var in unordered:
        df[var] = df[var].astype('string').str.lower()
        df[var] = recode_categories(
            df[var], replacement_mask=unordered_masks[var], min_frequency=unordered_min_frequencies[var]
        )
        df[var] = pd.Categorical(df[var], ordered=False)
    for var in numerical:
        df[var] = df[var].astype('float')
    return df


# In[90]:


def reduce_to_full_rank(A, tol=1e-10):
    """
    extract a full-rank matrix from A that has same rank as A, by dropping collinear columns
    
    Args: 
        - A (2D array)
    
    Returns: 
        - 2D array: a submatrix extracted from A
    """
    Q, R = np.linalg.qr(A) #QR decomposition
    independent = np.abs(np.diag(R)) > tol
    return A[:, independent]



# In[91]:


def get_non_collinear_instruments(Z, X_exog = None, tol=1e-10):
    """
    extract a full-rank matrix from Z whose columns are not collinear with other columns in Z and X_exog
    
    Args:
        - Z (2D array): e.g. matrix of instruments
        - X_exog (2D array): e.g. matrix of included exogenous variables
        - tol (float): the tolerance used to check if the regression SSR is equal to zero
    Returns:
        - 2D array: a submatrix extracted from Z

    Notes:
        - The idea is to regress iteratively each column from Z on other columns in Z and columns in X_exog, and suppress the dependant column from Z if there is a perfect fit
    """

    #Initialize the matrix X of regressor columns with X_exog (and include the constant column)
    if np.all(X_exog==None):
        X = np.ones((Z.shape[0], 1))
    else:
        X = np.asarray(X_exog)
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    keep = []
    NewZ = Z
    for col in Z.columns:
        #get the dependant column
        y = Z[col].values 
        #update the matrix of regressor columns with remaining columns in Z
        Xcol0 = np.hstack([NewZ.drop([col],axis=1), X])
        #get full-rank version of the matrix of regressor columns
        Xcol = reduce_to_full_rank(Xcol0)
        #fit depenant column on regressor columns
        beta = np.linalg.lstsq(Xcol, y, rcond=None)[0]
        y_hat = Xcol @ beta
        residual = y - y_hat    
        if np.linalg.norm(residual) > tol:
            keep.append(col)
        else:
            NewZ = NewZ.drop([col],axis=1)

    return Z[keep], keep


# In[93]:


def compute_rival_features(data, vars, marketvar='marketid', twodegree_polynomial_basis=False):
    """
    compute sum of polynomial basis functions of rival products' characteristics in vars.
    """
    
    df = data.copy()
    X = df[vars].values
    base = pd.DataFrame(index=df.index)

    # Basic group sums
    group_sum = df.groupby(marketvar)[vars].transform('sum')
    group_size = df.groupby(marketvar)[vars[0]].transform('count')
    base['Nb_RivalProducts'] = group_size - 1

    for i, var in enumerate(vars):
        base[f'{var}_RivalProducts'] = group_sum[var] - df[var]

    if twodegree_polynomial_basis:
        # Step 1: compute outer products for all rows
        # Result: (n_samples, n_combinations)
        combs = list(combinations_with_replacement(range(len(vars)), 2))
        n_combs = len(combs)
        prod_matrix = np.empty((len(df), n_combs))

        for k, (i, j) in enumerate(combs):
            prod_matrix[:, k] = X[:, i] * X[:, j]

        # Step 2: attach to DataFrame for groupby
        prod_df = pd.DataFrame(prod_matrix, columns=[f'{vars[i]}*{vars[j]}' for i, j in combs], index=df.index)
        prod_df[marketvar] = df[marketvar].values

        # Step 3: groupby sum once
        group_prod_sum = prod_df.groupby(marketvar).transform('sum')
        prod_df.drop(columns=marketvar, inplace=True)

        # Step 4: rival = group sum - own value
        rival_prod = group_prod_sum - prod_df
        # Rename columns
        rival_prod.columns = [col + '_RivalProducts' for col in rival_prod.columns]

        base = pd.concat([base, rival_prod], axis=1)

    return base


# In[94]:


def aggregate_data(data, pop_data, marketvar='marketid', productvar= 'make', twodegree_polynomial_instruments=False, 
                   aggfunc='mean', numerical=[], categorical=[], dep=[], endog=[], exog=[]):
    """
    aggregate the data at (market, product)-level

    Args:
        - data (pd.DataFrame): dataset of cars
        - pop_data: dataset of households sizes; contains at least following columns: 'state' and 'households (2015)'
        - marketvar, productvar (str): column names for market, product
        - twodegree_polynomial_instruments (bool): set value 'True' to obtain BLP instruments from 2-degree polynomial basis of exogenous characteristics
        - aggfunc (str): the aggregation function (preferably 'mean' or 'median')
        - numerical, categorical (list of str): the names of data columns that are numerical, categorical
        - dep, endog, exog (list of str): the names of data columns that are set as dependant variables, endogenous explanatory variables, enxogenous explanatory variables
    
    Returns : 
        - pd.DataFrame
        - list: names of dependant variables or variables in the returned dataframe that can be used to compute dependant variables 
        - list: names of endogenous explanatory variables in the returned dataframe
        - list: names of exogenous explanatory variables in the returned dataframe
        - list: names of excluded instruments in the returned dataframe

    Notes: 
        - the returned excluded instruments are the BLP instruments for price (i.e. sum of rival products' characteristics). see Berry, Levingson and Pakes, 1995.
        - 'state' column is needed inside data as key to get market sizes (proxied by state-varying households sizes of year 2015) from the pop_data dataframe
    """
    
    #Extract categories for each categorical variable (the most frequent category is excluded)
    Categories = [(var, cat) for var in categorical for cat in data[var].value_counts().index[1:]]

    df = data[[marketvar,productvar,'state']+numerical].copy(); pop_df = pop_data.copy()
    depvars = []; endogvars = []; exogvars = []
    for var in numerical:
        if var in dep:
            depvars = depvars + [var]
        if var in endog:
            endogvars = endogvars + [var]
        if var in exog:
            exogvars = exogvars + [var]
    for var, cat in Categories:
        varcat = f"{var}_{cat}"
        df[varcat] = (data[var] == cat).astype(float)
        if var in dep:
            depvars = depvars + [varcat]
        if var in endog:
            endogvars = endogvars + [varcat]
        if var in exog:
            exogvars = exogvars + [varcat]
    df = df.groupby([marketvar, productvar,'state'], observed=True).agg(aggfunc).reset_index()
    df['sales'] = data.groupby([marketvar, productvar,'state'], observed=True).size().values

    pop_df['households (2015)'] = pop_df['households (2015)'].map(lambda x: x.replace(',','')).astype('float')
    outsideoption_df = df[[marketvar,'state','sales']].groupby(by=[marketvar,'state'],observed=True).sum().reset_index().rename({'sales':'allsales'},axis=1)
    pop_df = pop_df.merge(outsideoption_df,on='state')
    df = pop_df.merge(df, on=[marketvar,'state'])
    df['share'] = df['sales']/df['households (2015)']
    df['share_oo'] = 1-(df['allsales']/df['households (2015)'])
    df['log_share_ratio'] = np.log(df['share']/df['share_oo'])

    depvars = depvars + ['households (2015)','allsales', 'share_oo', 'sales', 'share', 'log_share_ratio']

    #Extract instruments
    Z = compute_rival_features(df[[marketvar]+exogvars], vars=exogvars, marketvar='marketid', twodegree_polynomial_basis=twodegree_polynomial_instruments)
 
    #reduce instruments to collinearity-proof instruments
    Z, instrvars = get_non_collinear_instruments(Z, df[exogvars])
    df = df[[marketvar,productvar,'state']+depvars+endogvars+exogvars].reset_index(drop=True).join(Z.reset_index(drop=True))  
    
    return df , depvars, endogvars, exogvars, instrvars

