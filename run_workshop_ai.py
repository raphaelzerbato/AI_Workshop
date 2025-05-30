# %%
from data_preprocessing.data_prepro_func import *
from data_preprocessing.init_data_prepro import *
from instrument_creation.init_instrument_creation import *
from instrument_creation.instrument_creation_functions import *

from models_IV import *


import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# %%
if __name__ == "__main__":
    # %%    
    """
    load data and config
    """
    # Load model config
    model_config_path = 'config/model_config.yaml'
    model_config = load_config(model_config_path)

    # Load data config
    data_config_path = 'config/data_config.yaml'
    data_config = load_config(data_config_path)

    # load car data
    cars_file_path = data_config['loading_path_data']['data_cars']
    cars_db        = load_data(cars_file_path)

    # load population data
    population_file_path = data_config['loading_path_data']['data_population']
    population_db        = load_data(population_file_path)

    # %%
    # preprocessing the data
    preprocess_df, endogenous_var, exogenous_var, added_depvar = init_data_preprocessing(data_config, cars_db)

    # %%
     # Extract instruments
    Z, instrument_vars = init_instrument_creation(preprocess_df, exogenous_var, data_config['instrument_creation']['twodegree_polynomial_instruments'])
    
    # %%
    # Finalize the dataframe
    final_df = preprocess_df[
        ['marketid', data_config['var_of_interest']['productvar'], 'state']
        + added_depvar
        + endogenous_var
        + exogenous_var
    ].reset_index(drop=True).join(Z.reset_index(drop=True))

    # %%
    # Preprocess data
    X_train, X_test, y_train, y_test = spliting_data(cars_db, 'sellingprice')
    
    # Train model
    model_name = 'xgboost'
    trained_model = dp.train_selected_model(model_name, X_train, y_train, **config['model_params'])

# %%
