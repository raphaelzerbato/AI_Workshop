import scipy.stats as stats
import pandas as pd
import numpy as np
from models_IV.OLS import LinearModel
import matplotlib.pyplot as plt

def delta_method_ratio_pvalue(var1:str, var2:str, params:pd.DataFrame, cov:pd.DataFrame) -> float:

    """
    Compute the p-value of a ratio statistic  params[var1]/params[var2] using the Delta Method approximation of its variance

    Arguments:
        - var1, var2 (string objects): names of the two statistics that have estimated values in params
        - params (pd.DataFrame): observed value of a real vector-valued estimator
        - cov (pd.DataFrame): observed value of the variance-covariance matrix of params
    Returns:
        - pvalue (float object)

    Note:
        - Delta method approximation of var_(a/b) =  (1 / b**2) * var_a + (a**2 / b**4) * var_b - (2 * a / b**3) * cov_ab
    """

    if var1==var2:
        pvalue=0
    else:
        # Compute the ratio and its standard error using the Delta Method
        a = params[var1]
        b = params[var2]
        var_a = cov.loc[var1, var1]
        var_b = cov.loc[var2, var2]
        cov_ab = cov.loc[var1, var2]    
        ratio = a / b
        var_ratio = (1 / b**2) * var_a + (a**2 / b**4) * var_b - (2 * a / b**3) * cov_ab
        se_ratio = var_ratio**0.5
    
        #  Compute the z-statistic and p-value
        
        z = (ratio - 1) / se_ratio  # Test if ratio == 1
        pvalue = 2 * (1 - stats.norm.cdf(abs(z)))

    return pvalue

def demand_evaluation(OLS_model, exogenous_prediction=None, price_variable='sellingprice'):

    """
    OLS-estimate a demand model with a simple logit specification, i.e.
    Log(S) - Log(S_o) = X'Betax + Betap*P + epsilon

    Arguments:
        - X_train (pd.DataFrame): matrix of regressors, including the 'sellingprice' column
        - Y_train (pd.Series): dependent variable column, Log(S) - Log(S_o)
        - exogenous_prediction (None or pd.Series): an exogenous prediction of the 'sellingprice' column
            - if pd.Series, then the 'sellingprice' column in X_train will be replaced by price_train_predicted before estimation
                - the resulting econometric method is a 2SLS where the price is endogenous
            - if None, then the 'sellingprice' column in X_train remains unchanged
                - the resulting econometric method is an OLS where the price is exogenous

    Returns:
        - model (statsmodels.regression.linear_model.RegressionResultsWrapper): the estimated model
        - coefs (pd.DataFrame): dataframe with
            - estimated coefficients and their p-values,
            - estimated marginal willingnesses to pay and their p-values
        - Y_train_pred (pd.Series): prediction of dependent column in training sample

    Note:
        - the willingness to pay associated with a product attribute k in X is the ratio between the coef of k and the coef of the 'sellingprice' column
    """
   
    if not np.all(exogenous_prediction == None):
        OLS_model.X_train[exogenous_prediction.name] = np.array(exogenous_prediction)

    OLS_model.train(OLS_model.X_train, OLS_model.y_train)

    Y_train_pred = OLS_model._predict(OLS_model.X_train, OLS_model.feature_cols)

    OLS_model.varcovar_mat = OLS_model.compute_covariance_matrix(
        OLS_model.X_train,
        OLS_model.y_train
    )

    wtp_pvalues = compute_wtp_pvalues(
        OLS_model, 
        price_variable=price_variable
    )

    OLS_model.compute_pvalues(
        OLS_model.X_train,
        OLS_model.y_train
    )
   # Get coefficients and intercept
    coef_series = pd.Series(OLS_model.model.coef_, index=OLS_model.X_train.columns)
    coefs_with_intercept = pd.concat([pd.Series({'Intercept': OLS_model.model.intercept_}), coef_series])
 
    # Merge coefs, WTP (willingness to pay) and their pvalues
    coefs = pd.concat([coefs_with_intercept, OLS_model.pvalues,  coefs_with_intercept/np.abs(coefs_with_intercept['sellingprice']), wtp_pvalues], axis=1)
    coefs.columns = ['coef','coef-pvalue','wtp','wtp-pvalue']

    return OLS_model, coefs, Y_train_pred


def demand_evaluation_LASSO(trained_LASSO_model, exogenous_prediction=None):

    """
    OLS-estimate a demand model with a simple logit specification, i.e.
    Log(S) - Log(S_o) = X'Betax + Betap*P + epsilon

    Arguments:
        - X_train (pd.DataFrame): matrix of regressors, including the 'sellingprice' column
        - Y_train (pd.Series): dependent variable column, Log(S) - Log(S_o)
        - price_train_predicted (None or pd.Series): an exogenous prediction of the 'sellingprice' column
            - if pd.Series, then the 'sellingprice' column in X_train will be replaced by price_train_predicted before estimation
                - the resulting econometric method is a 2SLS where the price is endogenous
            - if None, then the 'sellingprice' column in X_train remains unchanged
                - the resulting econometric method is an OLS where the price is exogenous

    Returns:
        - model (statsmodels.regression.linear_model.RegressionResultsWrapper): the estimated model
        - coefs (pd.DataFrame): dataframe with
            - estimated coefficients and their p-values,
            - estimated marginal willingnesses to pay and their p-values
        - Y_train_pred (pd.Series): prediction of dependent column in training sample

    Note:
        - the willingness to pay associated with a product attribute k in X is the ratio between the coef of k and the coef of the 'sellingprice' column
    """
    
    X_train_new = trained_LASSO_model.X_train.copy()
    y_train_new = trained_LASSO_model.y_train.copy()

    if not np.all(exogenous_prediction == None):
        X_train_new[exogenous_prediction.name] = exogenous_prediction

    training_features = X_train_new.colnames

    IV2_steps = LinearModel(
        target_col='log_share_ratio',
        feature_cols = training_features,
        test_size = 0.2,
        random_state = 42
    )
    
    IV2_steps.X_train = X_train_new
    IV2_steps.y_train = y_train_new
    
    IV2_steps.train(
        IV2_steps.X_train,
        IV2_steps.y_train
    )
   # %% 
    Y_train_pred = IV2_steps._predict(
        IV2_steps.X_test, 
        IV2_steps.feature_cols
    )

    wtp_pvalues = compute_wtp_pvalues(
        IV2_steps, 
        target_variable='sellingprice'
    )

    IV2_steps.compute_pvalues(
        IV2_steps,
        IV2_steps.X_train,
        IV2_steps.y_train
    )
    # Get coefficients and intercept
    coef_series = pd.Series(IV2_steps.model.coef_, index=IV2_steps.X_train.columns)
    coefs_with_intercept = pd.concat([pd.Series({'Intercept': IV2_steps.model.intercept_}), coef_series])
 
    # Merge coefs, WTP (willingness to pay) and their pvalues
    coefs = pd.concat([coefs_with_intercept, IV2_steps.pvalues,  coefs_with_intercept/np.abs(coefs_with_intercept['sellingprice']), wtp_pvalues], axis=1)
    coefs.columns = ['coef','coef-pvalue','wtp','wtp-pvalue']
 
    return IV2_steps, coefs, Y_train_pred


def compute_wtp_pvalues(linear_model:LinearModel, price_variable='sellingprice') -> pd.Series:
    # il faut que je change target variable pour que cela soit un attribut
    """
    Compute the p-values of the willingness to pay (WTP) for each product attribute in coefs

    Arguments:
        - coefs (pd.DataFrame): dataframe with estimated coefficients and their p-values
        - cov (pd.DataFrame): variance-covariance matrix of the estimated coefficients

    Returns:
        - WTP_pvalue (pd.Series): series with p-values of the WTP for each product attribute
    
    Note:
        - The linear_model.varcovar_mat should have been computed before calling this function
    """
    #### NEEDS TO BE EXTENDED TO HANDLE OTHER TYPES OF OLS MODELS FROM STATS MODELS ####
    if linear_model.varcovar_mat is None or linear_model.varcovar_mat.empty:
        linear_model.compute_covariance_matrix(   
            linear_model.X_train,
            linear_model.y_train
        )
    # Get coefficients and intercept
    coefs = pd.Series(linear_model.model.coef_, index=linear_model.X_train.columns)
    coefs_with_intercept = pd.concat([pd.Series({'Intercept': linear_model.model.intercept_}), coefs])
 
    cov = linear_model.varcovar_mat
    WTP_pvalue = pd.Series(index=coefs_with_intercept.index, dtype=float)

    for var in coefs.index:
        if var != price_variable:
            WTP_pvalue[var] = delta_method_ratio_pvalue(var, price_variable, coefs, cov)

    return WTP_pvalue

def plot_coefs(coef_series, pvalue_series, titlename):

    """
    Shows a graph, in annotated bar plots, of observed values of a vector-valued statistic with the corresponding pvalues
    This function is used to visualize the coefficients of a linear model and their p-values,
    It's also used to visualize wtp statistics and their p-values
    """
  
    # pvalue_stars
    pvalue_stars_series = pvalue_series.map(
        lambda x: '****' if x<0.001 else ('***' if x<0.01 else ('  **' if x<0.05 else ('   *' if x<0.1 else '      ')))
    )

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot the bars with color based on value
    bars = ax.bar(coef_series.index, [0.1 if v>=0 else -0.1 for v in coef_series],  # All bars have the same length
                  color=['#4c72b0' if v >= 0 else '#c44e52' for v in coef_series])
    
    # Add true value labels on top of the bars
    ax.bar_label(
        bars, labels=[f'{coef_series[name]:.3f} {pvalue_stars_series[name]}' for name in coef_series.index],
        padding=3, fontsize=10, rotation=90
    )
    
    # Add title and labels
    ax.set_title(f"{titlename} \n\n(****: pvalue < 0.001, ***: pvalue < 0.01, **: pvalue < 0.05, *: pvalue < 0.1)\n")
    ax.set_xlabel("Product Attribute")
    ax.set_ylabel("")
    ax.axhline(0, color='black', linewidth=0.8)
    
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    
    # Remove y ticks
    ax.set_yticks([])
    
    ax.set_ylim((-.22,.22))
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    plt.show()