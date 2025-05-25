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
from itertools import product, combinations_with_replacement
import yaml


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

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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

def define_market(data, market_vars, market_label='market', market_code='marketid', minsize=20, DropSmallMarkets=True) -> pd.DataFrame:
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
    
    if DropSmallMarkets:
        selected_markets = markets_tab[markets_tab>=minsize].index
        nb_droppedmarkets = np.sum(~(markets_tab>=minsize))
        nb_droppedrows = np.sum(~df['market'].isin(selected_markets))
        df = df[df['market'].isin(selected_markets)]
        print(f"\n{nb_droppedmarkets} markets with sizes lower than {minsize} have been dropped making {nb_droppedrows} dropped rows \n")
    else:
        selected_markets = markets_tab.index
    
    idmarket_mask = dict([(i, selected_markets[i]) for i in range(len(selected_markets))])
    marketid_mask = dict([(selected_markets[i], i) for i in range(len(selected_markets))])
    
    df['marketid'] = df['market'].map(marketid_mask)
    return df

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
    s = s.astype('string').str.lower()
    s = s.replace(replacement_mask)
    frequencies = s.value_counts()
    popular_values = set(frequencies[frequencies>= min_frequency].index)-{''}
    s = s.map(lambda x: x if pd.isna(x) or x in popular_values else 'other')
    return s
 
def order_unorder_categorical_var(data, ordered=[], unordered=[], 
                               ordered_masks=dict(), unordered_masks=dict(), 
                               ordered_min_frequencies=dict(), unordered_min_frequencies=dict()) -> pd.DataFrame:
    """
    Extract and process relevant variables
    Args:
        - data (pd.DataFrame): should contain a 'state' column
        - ordered, unordered (list of str): the names of relevant columns that will be set as ordered categorical and unordered categorical 
        - ordered_masks, unordered_masks (dict of dict): dictionaries of replacement masks to be used as arguments in the recode_categories function for each relevant categorical variable
        - ordered_min_frequencies, unordered_min_frequencies (dict of int): dictionaries that provide the min_frequency argument for the recode_categories function for each relevant categorical variable

    Returns:
        - pd.DataFrame: contains the provided relevant columns along with marketvar and productvar columns
    """
    df = data.copy()    

    for var in ordered:
        df[var] = recode_categories(
            df[var], replacement_mask=ordered_masks[var], min_frequency=ordered_min_frequencies[var]
        )
        df[var] = pd.Categorical(df[var], ordered=True)

    for var in unordered:
        df[var] = recode_categories(
            df[var], replacement_mask=unordered_masks[var], min_frequency=unordered_min_frequencies[var]
        )
        df[var] = pd.Categorical(df[var], ordered=False)
    return df

def one_hot_encoder(data, categorical, exog, endog, ):
    """
    One-hot encode categorical variables and updates the list of
    endogenous and exogenous variables with the dummies and removes the main category keeping track 
    of it with excluded_main_categories.
    """
    df = data.copy()
    excluded_main_categories = []
    for var in categorical:
        # Identify the most populated category for the variable
        main_category = data[var].value_counts().idxmax()
        excluded_main_categories.append(main_category)
        
        # One-hot encode the variable
        encoded_df = pd.get_dummies(data[var], prefix=var, drop_first=False)

        # Remove the column corresponding to the main category
        main_category_column = f"{var}_{main_category}"

        encoded_df = encoded_df.drop(columns=[main_category_column])

        # Add the remaining encoded columns to the main dataframe
        df = pd.concat([df, encoded_df], axis=1)
        
        # Update exog and endog variables
        dummy_var = list(encoded_df.columns)
        if var in exog:
            exog.remove(var)
            exog = exog + dummy_var
        if var in endog:   
            endog.remove(var)
            endog = endog + dummy_var
            
    return df, endog, exog, excluded_main_categories

def preprocess_color_interior(df):
    """
    Preprocess the 'color' and 'interior' columns in the DataFrame.
    """
    for var in ['color', 'interior']:
        if var in df.columns:
            df[var] = df[var].replace({'—':pd.NA})
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
    df = data.copy()
    var_list = var_of_interest.copy()
    for var in [marketvar, productvar]:
        if var in var_of_interest:
            var_list.remove(var)
    df = df[[marketvar, productvar] + var_list]
    print('\nNumber of missing per relevant variable in the remaining dataframe \n', np.sum(df.isna(), axis=0))
    if DropNa:
        print(f"\nA total of {np.sum(np.any(df.isna(), axis=1))} rows with missing data in relevant variables have been dropped \n")
        df = df.dropna()
    return df


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

def compute_market_shares(df, pop_df, marketvar, productvar):
    """
    Compute market shares and related variables.
    """
    outsideoption_df = df[[marketvar, 'state', 'sales']].groupby(
        by=[marketvar, 'state'], observed=True
    ).sum().reset_index().rename({'sales': 'allsales'}, axis=1)
    pop_df = pop_df.merge(outsideoption_df, on='state')
    df = pop_df.merge(df, on=[marketvar, 'state'])
    df['share'] = df['sales'] / df['households (2015)']
    df['share_oo'] = 1 - (df['allsales'] / df['households (2015)'])
    df['log_share_ratio'] = np.log(df['share'] / df['share_oo'])
    return df


def extract_instruments(df, exogvars, marketvar, twodegree_polynomial_instruments):
    """
    Extract instruments for the model:
        - compute sum of polynomial basis functions of rival products' characteristics in exogvars.
    """

    X = df[exogvars].values
    Z = pd.DataFrame(index=df.index)

    # Basic group sums
    group_sum = df.groupby(marketvar)[exogvars].transform('sum')
    group_size = df.groupby(marketvar)[exogvars[0]].transform('count')
    Z['Nb_RivalProducts'] = group_size - 1

    for i, var in enumerate(exogvars):
        Z[f'{var}_RivalProducts'] = group_sum[var] - df[var]

    if twodegree_polynomial_instruments:
        # Step 1: compute outer products for all rows
        # Result: (n_samples, n_combinations)
        combs = list(combinations_with_replacement(range(len(exogvars)), 2))
        n_combs = len(combs)
        prod_matrix = np.empty((len(df), n_combs))

        for k, (i, j) in enumerate(combs):
            prod_matrix[:, k] = X[:, i] * X[:, j]

        # Step 2: attach to DataFrame for groupby
        prod_df = pd.DataFrame(prod_matrix, columns=[f'{exogvars[i]}*{exogvars[j]}' for i, j in combs], index=df.index)
        prod_df[marketvar] = df[marketvar].values

        # Step 3: groupby sum once
        group_prod_sum = prod_df.groupby(marketvar).transform('sum')
        prod_df.drop(columns=marketvar, inplace=True)

        # Step 4: rival = group sum - own value
        rival_prod = group_prod_sum - prod_df
        # Rename columns
        rival_prod.columns = [col + '_RivalProducts' for col in rival_prod.columns]

        Z = pd.concat([Z, rival_prod], axis=1)
    return Z


def remove_duplicates_keep_order(lst):
    """
    Remove duplicates from a list of strings while keeping the order
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def aggregate_data(data, pop_data, marketvar='marketid', productvar='make', twodegree_polynomial_instruments=False,
                   aggfunc='mean', numerical=[], categorical=[], dep=[], endog=[], exog=[]):
    """
    Aggregate the data at (market, product)-level.
    """
    #initialize variables
    init_vars = remove_duplicates_keep_order([marketvar, productvar, 'state'] + dep + endog + exog)
    df = data[init_vars].copy()
    pop_df = pop_data.copy()

    depvars = dep
    
    # Extract categories
    results = one_hot_encoder(df, categorical, exog, endog)
    df = results[0]; endogvars = results[1]; exogvars = results[2]
    excluded_main_categories = results[3]

    # Drop columns that have corresponding one-hot-encodings (keep the state and make columns)
    df = df.drop(columns=set(categorical)-{'state','make'})

    # Extract the variables that we need
    df = df[[marketvar, productvar, 'state'] + dep + endogvars + exogvars]

    # Aggregate data and compute sales
    df = df.groupby([marketvar, productvar, 'state'], observed=True).agg(aggfunc).reset_index()
    df['sales'] = data.groupby([marketvar, productvar, 'state'], observed=True).size().values

    # Process households data and compute market shares
    pop_df['households (2015)'] = pop_df['households (2015)'].astype('string').str.replace(',', '').astype('float')
    df = compute_market_shares(df, pop_df, marketvar, productvar)

    # Update dependent variables
    depvars += ['households (2015)', 'allsales', 'share_oo', 'sales', 'share', 'log_share_ratio']

    # Extract instruments
    Z = extract_instruments(df, exogvars, marketvar, twodegree_polynomial_instruments)

    # Reduce instruments to collinearity-proof instruments
    Z, instrvars = get_non_collinear_instruments(Z, df[exogvars])

    # Finalize the dataframe
    df = df[[marketvar, productvar, 'state'] + depvars + endogvars + exogvars].reset_index(drop=True).join(Z.reset_index(drop=True))

    return df, depvars, endogvars, exogvars, instrvars, excluded_main_categories    


    
