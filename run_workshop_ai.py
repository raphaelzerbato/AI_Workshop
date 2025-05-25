# %%
from preprocessing.initiate_preprocessing import init_preprocessing
from preprocessing.preprocessing_functions import load_data, load_config
from instrument_creation.init_instrument_creation import init_instrument_creation
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
    # preprocessing the data
    preprocess_df, endogenous_var, exogenous_var, added_depvar = init_preprocessing(data_config, cars_db)

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
    X_train, X_test, y_train, y_test = splitting_data(cars_db, 'price')
    
    # Train model
    model_name = 'xgboost'
    trained_model = dp.train_selected_model(model_name, X_train, y_train, **config['model_params'])

