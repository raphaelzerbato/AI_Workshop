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

def preprocess_color_interior(df):
    """
    Preprocess the 'color' and 'interior' columns in the DataFrame.
    """
    for var in ['color', 'interior']:
        if var in df.columns:
            df[var].replace({'—':np.nan})  
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

    # bug ici
    outsideoption_df = df[['marketid','state','sales']].groupby(
        by=['marketid', 'state'], observed=True
    ).sum().reset_index().rename({'sales': 'allsales'}, axis=1)

    pop_df = load_prepro_pop_data(data_config)

    pop_df = pop_df.merge(outsideoption_df,on='state')
    df = pop_df.merge(df, on=['marketid', 'state'])
    
    df['share'] = df['sales'] / df['population (2015)']
    df['share_oo'] = 1 - (df['allsales'] / df['population (2015)'])
    df['log_share_ratio'] = np.log(df['share'] / df['share_oo'])
    
    return df


def compute_market_shares(df, pop_df, marketid, productvar):
    """
    Compute market shares and related variables.
    """
    outsideoption_df = df[[marketid, 'state', 'sales']].groupby(
        by=[marketid, 'state'], observed=True
    ).sum().reset_index().rename({'sales': 'allsales'}, axis=1)
    pop_df = pop_df.merge(outsideoption_df, on='state')
    df = pop_df.merge(df, on=[marketid, 'state'])
    df['share'] = df['sales'] / df['population (2015)']
    df['share_oo'] = 1 - (df['allsales'] / df['population (2015)'])
    df['log_share_ratio'] = np.log(df['share'] / df['share_oo'])
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
    df['share'] = df['sales'] / df['population (2015)']
    df['share_oo'] = 1 - (df['allsales'] / df['population (2015)'])
    df['log_share_ratio'] = np.log(df['share'] / df['share_oo'])
    return df


def extract_instruments(df, exogvars, marketvar, twodegree_polynomial_instruments):
    """
    Extract instruments for the model.
    """
    if not twodegree_polynomial_instruments:
        Z = df[[marketvar] + exogvars].groupby([marketvar]).apply(
            lambda x: x.assign(**dict(
                [('Nb_RivalProducts', x.shape[0] - 1)] +
                [(var + '_RivalProducts', x[var].sum() - x[var]) for var in exogvars]
            )), include_groups=False
        ).reset_index(drop=True).drop(exogvars, axis=1)
    else:
        Z = df[[marketvar] + exogvars].groupby([marketvar]).apply(
            lambda x: x.assign(**dict(
                [('Nb_RivalProducts', x.shape[0] - 1)] +
                [(var + '_RivalProducts', x[var].sum() - x[var]) for var in exogvars] +
                [(var1 + '*' + var2 + '_RivalProducts', (x[var1] * x[var2]).sum() - x[var1] * x[var2])
                 for var1, var2 in list(product(exogvars, repeat=2))]
            )), include_groups=False
        ).reset_index(drop=True).drop(exogvars, axis=1)
    return Z


def aggregate_data(data, pop_data, marketvar='marketid', productvar='make', twodegree_polynomial_instruments=False,
                   aggfunc='mean', numerical=[], categorical=[], dep=[], endog=[], exog=[]):
    """
    Aggregate the data at (market, product)-level.
    """
    df = data[[marketvar, productvar, 'state'] + numerical].copy()
    pop_df = pop_data.copy()
    # Extract categories and initialize variables
    excluded_main_categories = []
    
    for var in categorical:
        results = one_hot_encoder(df, var, exog, endog)
        df = results[0]; endogvars = endogvars + results[1]; exogvars = exogvars + results[2]
        excluded_main_categories = excluded_main_categories.append(results[3])  

    # Aggregate data and compute sales
    df = df.groupby([marketvar, productvar, 'state'], observed=True).agg(aggfunc).reset_index()
    df['sales'] = data.groupby([marketvar, productvar, 'state'], observed=True).size().values

    # Process population data and compute market shares
    pop_df['population (2015)'] = pop_df['population (2015)'].map(lambda x: x.replace(',', '')).astype('float')
    df = compute_market_shares(df, pop_df, marketvar, productvar)

    # Update dependent variables
    depvars += ['population (2015)', 'allsales', 'share_oo', 'sales', 'share', 'log_share_ratio']

    # Extract instruments
    Z = extract_instruments(df, exogvars, marketvar, twodegree_polynomial_instruments)

    # Reduce instruments to collinearity-proof instruments
    Z, instrvars = get_non_collinear_instruments(Z, df[exogvars])

    # Finalize the dataframe
    df = df[[marketvar, productvar, 'state'] + depvars + endogvars + exogvars].reset_index(drop=True).join(Z.reset_index(drop=True))

    return df, depvars, endogvars, exogvars, instrvars, excluded_main_categories    


    
