# %%
import preprocessing.preprocessing_car_data as dp
from preprocessing.preprocessing_car_data import *
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# %%
if __name__ == "__main__":

    # %%    
    """
    load data and config
    """

    # Load config
    model_config_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/config/model_config.yaml'
    model_config = load_config(model_config_path)

    data_config_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/config/data_config.yaml'
    # Load data config
    data_config = load_config(data_config_path)

    # load data
    cars_file_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/data/data_cars/car_prices.csv'
    cars_db        = load_data(cars_file_path)

    # %%
    """
    process dates and define markets
    """
    # Process dates 
    df = dp.process_dates(cars_db)

    # Define markets
    df = dp.define_market(df, 
        market_vars = data_config['var_of_interest']['marketvar'],
        minsize = data_config['minisize_of_market'])

    # preprocess color and interior
    df = dp.preprocess_color_interior(df)    

    # exctract variables of intrest
    df = dp.subset_var_of_interest(df,
    data_config['var_of_interest']['numerical'] + data_config['var_of_interest']['unordered'], 
    marketvar = 'marketid', productvar = data_config['var_of_interest']['productvar'])
    
    # preprocess color and interio
    df = dp.preprocess_color_interior(df)

    # %%
    # preprocess categorical variables
    df = dp.treat_categories(
        df, 
        variable = data_config['var_of_interest']['ordered'],
        masks=data_config['ordered_masks'],
        min_frequencies=data_config['ordered_min_frequencies'], 
        order_serie = True
        )

    df = dp.treat_categories(
        df,
        variable=data_config['var_of_interest']['unordered'],
        masks=data_config['unordered_masks'], 
        min_frequencies=data_config['unordered_min_frequencies'], 
        order_serie = False
        )
        
    # %%
    # one hot encoding of the categorical variables
    exo_cats, endo_cats = dp.get_categorical_intersections(data_config)
    
    df_one_hot, added_exo_dummies, excluded_exo_main = dp.one_hot_encoder(df, exo_cats)
    df_one_hot, added_endo_dummies, excluded_endo_main = dp.one_hot_encoder(df_one_hot, endo_cats)

    endogenous_var = update_list(data_config['var_of_interest']['endog'],
                                added_endo_dummies,
                                endo_cats)
    
    exogenous_var = update_list(data_config['var_of_interest']['exog'],
                                added_exo_dummies,
                                exo_cats)
    
    # %%
    df_one_hot, added_depvar = compute_sales_marketshare(df_one_hot, data_config, aggfunc='mean')

    # %%
     # Extract instruments
    Z = extract_instruments(df_one_hot, exogenous_var,
                            marketvar = 'marketid',
                            twodegree_polynomial_instruments=False)

    # Reduce instruments to collinearity-proof instruments
    Z, instrvars = get_non_collinear_instruments(Z, df_one_hot[exogenous_var])

    # Finalize the dataframe
    final_df = df_one_hot[['marketid', data_config['var_of_interest']['productvar'], 'state'] + added_depvar + endogenous_var + exogenous_var].reset_index(drop=True).join(Z.reset_index(drop=True))
    

    # %%
    # Preprocess data
    X_train, X_test, y_train, y_test = splitting_data(cars_db, 'price')
    
    # Train model
    model_name = 'xgboost'
    trained_model = dp.train_selected_model(model_name, X_train, y_train, **config['model_params'])

