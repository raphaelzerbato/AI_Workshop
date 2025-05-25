"""
all key functions for instrument creation
"""
import numpy as np
import pandas as pd
from itertools import product

def extract_instruments(df, exogvars, marketvar, twodegree_polynomial_instruments):
    """
    Extract instruments for the model.
    The number of other products in the same market.

    The sum of exogenous variables across other products in the market (excluding the current row).

    The sum of all pairwise interactions between exogenous variables across other products (excluding the current row).

    # exogvasr = ['odometer', 'condition', 'interior_other', 'interior_gray', 'interior_tan',
    'interior_beige', 'make_chevrolet', 'make_nissan', 'make_toyota','make_chrysler', 'make_bmw',..., 'year_..', 'make_', 
    'color_', 'body_']
    market_var = 'marketid'
    """
    # Check if the variable `twodegree_polynomial_instruments` is not set or is falsy
    if not twodegree_polynomial_instruments:
        # Create a new DataFrame `Z` by grouping `df` by `marketvar` and applying a transformation
        Z = df[[marketvar] + exogvars].groupby([marketvar]).apply(
            # For each group, assign new columns:
            lambda x: x.assign(
                # Add a column `Nb_RivalProducts` which is the count of rows in the group minus 1
                **dict(
                    [('Nb_RivalProducts', x.shape[0] - 1)] +
                    # For each variable in `exogvars`, add a column `<var>_RivalProducts`
                    # which is the sum of the variable in the group minus the value for the current row
                    [(var + '_RivalProducts', x[var].sum() - x[var]) for var in exogvars]
                )
            ),
            # `include_groups=False` is intended to exclude the grouping columns from the output
            include_groups=False
        ).reset_index(drop=True).drop(exogvars, axis=1)     # Reset the index of the resulting DataFrame
        # Drop the original `exogvars` columns from the DataFrame `Z`
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

### im here
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
        - The idea is to regress iteratively each column from Z on other columns in Z and columns in X_exog, 
        and suppress the dependant column from Z if there is a perfect fit
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