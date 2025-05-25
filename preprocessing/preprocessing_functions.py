"""
Functions for loading and preprocessing car data.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, Input
import statsmodels.api as sm
#from linearmodels.iv import IV2SLS
import seaborn as sns
from itertools import product
import yaml

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

def define_market(data, market_vars, minsize=20) -> pd.DataFrame:
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

    df = df[df['market'].isin(selected_markets)]

    print(f"\n{nb_droppedmarkets} markets with sizes lower than {minsize} have been dropped making {nb_droppedrows} dropped rows \n")

    idmarket_mask = dict([(i, selected_markets[i]) for i in range(len(selected_markets))])
    marketid_mask = dict([(selected_markets[i], i) for i in range(len(selected_markets))])
    
    df['marketid'] = df['market'].map(marketid_mask)
    return df


def recode_categories(series:pd.Series, replacement_mask=dict(), min_frequency=10000, ordered_serie= True) -> pd.DataFrame:
    """
    recode values of a series of strings
    Args:
        - data (pd.DataFrame): dataframe containing the series to be recoded
        - var (str): name of the column to be recoded
        - replacement_mask (dict): a dictionary that maps some values to be replaced to their new values
        - min_frequency (int): minimum frequency per category to allow
                        all values with a lower frequency will be recoded as 'other'
    Returns: pd.DataFrame
    """
    s = series.copy()     
    s = s.replace(replacement_mask)
    s = s.map(lambda x: str(x).lower())
    frequencies = s.value_counts()

    popular_values = set(frequencies[frequencies>= min_frequency].index)-{''}
    s = s.map(lambda x: x if x in popular_values else 'other')
    s = pd.Categorical(s, ordered = ordered_serie)
    return s
    

def treat_categories(data, variable=[], masks=dict(), min_frequencies=dict(), order_serie=True) -> pd.DataFrame:
    """
    Extract and process relevant variables
    Args:
        - data (pd.DataFrame): should contain a 'state' column
        - variable (list of str): the names of relevant columns to process
        - masks (dict of dict): dictionaries of replacement masks to be used as arguments in the recode_categories function for each relevant categorical variable
        - min_frequencies (dict of int): dictionaries that provide the min_frequency argument for the recode_categories function for each relevant categorical variable
        - order_serie (bool): whether to set the resulting categories as ordered

    Returns:
        - pd.DataFrame: contains the provided relevant columns along with marketvar and productvar columns
    """
    if not variable:  # Check if variable list is empty
        return data  # Do nothing and return the original data

    df = data.copy()
    # Extract relevant variables
    for var in variable:
        df[var] = recode_categories(
            df[var], replacement_mask=masks.get(var, {}), min_frequency=min_frequencies.get(var, 0), ordered_serie=order_serie
        )

    return df

def get_categorical_intersections(config):
    """
    Extract the intersection of categorical variables with exogenous and endogenous variables.

    Args:
        config (dict): Configuration dictionary containing 'var_of_interest' with keys 'categorical', 'exog', and 'endog'.

    Returns:
        tuple: Two lists - exogenous categorical variables and endogenous categorical variables.
    """
    var_of_interest = config.get('var_of_interest', {})
    cat_vars = set(var_of_interest.get('categorical', []))
    exo_cats = list(cat_vars.intersection(var_of_interest.get('exog', [])))
    endo_cats = list(cat_vars.intersection(var_of_interest.get('endog', [])))
    return exo_cats, endo_cats

def one_hot_encoder(data, categorical):#, exog, endog):
    """
    One-hot encode categorical variables and updates the list of
    endogenous and exogenous variables with the dummies and removes the main category keeping track 
    of it with excluded_main_categories.
    """
    df = data.copy()
    excluded_main_categories = []
    added_dummies = []
    for var in categorical:
        # Identify the most populated category for the variable
        main_category = data[var].value_counts().idxmax()
        excluded_main_categories.append(main_category)
        # One-hot encode the variable
        encoded_df = pd.get_dummies(df[var], prefix=var, drop_first=False)

        # Remove the column corresponding to the main category
        main_category_column = f"{var}_{main_category}"
        encoded_df = encoded_df.drop(columns=[main_category_column])

        # Add the remaining encoded columns to the main dataframe
        df = pd.concat([df, encoded_df], axis=1)
        
        # remove the original categorical column
        if var not in ['make']:
            df = df.drop(columns=[var])
    
        # Update exog and endog variables
        dummy_var = list(set(df.columns) - set(data.columns))
        
        added_dummies.extend(dummy_var)

    return df, added_dummies, excluded_main_categories

def update_list(lst, new_items_add, item_to_remove=None):
    """
    Update a list by adding new items and optionally removing an item.
    
    Args:
        lst (list): The original list to be updated.
        new_items_add (list): Items to be added to the list.
        item_to_remove (str, optional): Item to be removed from the list. Defaults to None.
        
    Returns:
        list: Updated list with new items added and specified item removed.
    """
    lst.extend(item for item in new_items_add if item not in lst)
    for item in item_to_remove:
        if item is not None and item in lst:
            lst.remove(item)
    return lst


def preprocess_color_interior(df):
    """
    Preprocess the 'color' and 'interior' columns in the DataFrame.
    """
    for var in ['color', 'interior']:
        if var in df.columns:
            color_table = df[var].value_counts()
            popular_colors = set(color_table[color_table>=10000].index)-{'—'}
            df[var] = df[var].map(lambda x: x if x in popular_colors else 'other')
    return df     

def change_to_numerical(data, numerical):
    """"
    Change the type of the variables in numerical to float
    """
    df = data.copy()
    for var in numerical:
        df[var] = df[var].astype('float')    
    return df

def subset_var_of_interest(data, var_of_interest, marketvar='marketid', productvar='make', DropNa=True):
    """
    Subset the data to keep only the variables of interest
    """
    # Ensure no duplicate names in the list
    relevant_vars = list(set([marketvar, productvar, 'state'] + var_of_interest))
    
    df = data.copy()
    print('\nNumber of missing per relevant variable in the remaining dataframe \n', np.sum(df.isna(), axis=0))
    print(f"\nA total of {np.sum(np.any(df.isna(), axis=1))} rows with missing data in relevant variables have been dropped \n")
    if DropNa:
        df = df.dropna()
    df = df[relevant_vars]
    return df

def load_prepro_pop_data(data_config):
    """
    Loads and preprocesses population data from a specified file path.

    This function reads population data from a file specified in the 
    `data_config` dictionary, processes the population column to remove 
    commas, and converts it to a float type for further analysis.

    Args:
        data_config (dict): A configuration dictionary containing the 
            file path to the population data under the key 
            'loading_path_data' -> 'data_population'.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed population 
        data with the 'population (2015)' column cleaned and converted 
        to float.

    Example:
        data_config = {
            'loading_path_data': {
                'data_population': 'path/to/population_data.csv'
            }
        }
        pop_df = load_prepro_pop_data(data_config)
    """
    population_file_path = data_config['loading_path_data']['data_population']
    pop_df   = load_data(population_file_path)
    pop_df['population (2015)'] = pop_df['population (2015)'].map(lambda x: x.replace(',','')).astype('float')
    return pop_df

def compute_sales_marketshare(data, data_config, aggfunc='mean'):
    """
    Compute market share and related variables.

    Args:
        data (pd.DataFrame): Input data containing market and product information.
        marketvar (str): Column name representing the market identifier.
        productvar (str): Column name representing the product identifier.
        aggfunc (str or function): Aggregation function to apply (default is 'mean').

    Returns:
        pd.DataFrame: DataFrame with aggregated data and computed sales.
    """
    
    df = data.copy()

    productvar = data_config['var_of_interest']['productvar']

    group_by_vars = ['marketid', productvar, 'state']

    df = df.groupby(group_by_vars, observed=True).agg(aggfunc).reset_index()

    df['sales'] = data.groupby(group_by_vars, observed=True).size().values

    outsideoption_df = df[['marketid','state','sales']].groupby(
        by=['marketid', 'state'], observed=True
    ).sum().reset_index().rename({'sales': 'allsales'}, axis=1)

    pop_df = load_prepro_pop_data(data_config)

    pop_df = pop_df.merge(outsideoption_df,on='state')
    df = pop_df.merge(df, on=['marketid', 'state'])
    
    df['share'] = df['sales'] / df['population (2015)']
    df['share_oo'] = 1 - (df['allsales'] / df['population (2015)'])
    df['log_share_ratio'] = np.log(df['share'] / df['share_oo'])
    
    depvars = ['population (2015)', 'allsales', 'share_oo', 'sales', 'share', 'log_share_ratio']
    return df, depvars


    
