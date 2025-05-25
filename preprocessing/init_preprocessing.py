import preprocessing.preprocessing_functions as dp
from preprocessing.preprocessing_functions import *

def init_preprocessing(data_config, cars_db):
    """
    Initialize preprocessing functions and configurations.
    This function is called at the start of the preprocessing pipeline and 
    calls all the functions from preprocessing.preprocessing_functions
    """
    # Initialize any necessary configurations or parameters here
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
    
    return df_one_hot, endogenous_var, exogenous_var, added_depvar

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