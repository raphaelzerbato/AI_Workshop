"""
Functions for loading and preprocessing car data.
""""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import seaborn as sns
from itertools import product



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

def define_market(data, market_vars, market_label='market', market_code='marketid', minsize=20) -> pd.DataFrame:
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

    print(f"\n{nb_droppedmarkets} markets with sizes lower than {minsize} have been dropped; 
    making {nb_droppedrows} dropped rows \n")

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
    s = s.replace(replacement_mask)
    s = s.map(lambda x: str(x).lower())
    frequencies = s.value_counts()
    popular_values = set(frequencies[frequencies>= min_frequency].index)-{''}
    s = s.map(lambda x: x if x in popular_values else 'other')
    return s

def order_unorder_categorical_var(data, dropna=True, ordered=[], unordered=[], 
                               ordered_masks=dict(), unordered_masks=dict(), 
                               ordered_min_frequencies=dict(), unordered_min_frequencies=dict()) -> pd.DataFrame:
    """
    Extract and process relevant variables
    Args:
        - data (pd.DataFrame): shouldcontaain a 'state' column
        - numerical, ordered, unordered (list of str): the names of relevant columns that will be set as numerical, ordered categorical and unordered categorical 
        - ordered_masks, unordered_masks (dict of dict): dictionaries of replacement masks to be used as arguments in the recode_categories function for each relevant categorical variable
        - ordered_min_frequencies, unordered_min_frequencies (dict of int): dictionaries that provides the min_frequency argument for the recode_categories function for each relevant categorical variable

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
            df[var], replacement_mask=ordered_masks[var], min_frequency=ordered_min_frequencies[var]
        )
        df[var] = pd.Categorical(df[var], ordered=False)
    return df

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

def subset_var_of_interest(data, var_of_interest, marketvar='marketid', productvar='make'):
    """
    Subset the data to keep only the variables of interest
    """
    df = data.copy()
    print('\nNumber of missing per relevant variable in the remaining dataframe \n', np.sum(df.isna(), axis=0))
    print(f"\nA total of {np.sum(np.any(df.isna(), axis=1))} rows with missing data in relevant variables have been dropped \n")
    if dropna:
        df = df.dropna()
    df = df[[marketvar, productvar] + var_of_interest]
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

