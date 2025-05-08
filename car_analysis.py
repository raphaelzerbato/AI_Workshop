# %%
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# %%

"""
Load data and configs function
"""
# Load  data from file
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

# Load config
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

# %% 
"""
Treating the data function
"""
def splitting_data(data:pd.dataframe, Y_var:str)->pd.dataframe:
    # Splitting the data into train and test sets
    X = data.drop(columns=[Y_var])
    y = data[Y_var]
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 
    

def train_selected_model(model_name, X_train, y_train, **kwargs):
    """
    Train a specified model on the provided data.

    Parameters:
        model_name (str): Name of the model ('xgboost', 'lightgbm', 'randomforest', 'catboost').
        X_train (array): Training features.
        y_train (array): Training labels.
        **kwargs: Additional parameters for the model.

    Returns:
        model: Trained model instance.
    """
    model_name = model_name.lower()
    
    if model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(**kwargs)
    elif model_name == 'randomforest':
        model = RandomForestClassifier(**kwargs)
    elif model_name == 'catboost':
        model = CatBoostClassifier(verbose=0, **kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    model.fit(X_train, y_train)
    return model

# %%
# Example usage
if __name__ == "__main__":
    
    # Load car data from a YAML file
    data = load_data('car_data.csv')
    
    # Load configuration from a YAML file
    config = load_config('config.yaml')

    # Splitting the data into train and test sets
    train_data, test_data = splitting_data(data,)

    # Example: get LightGBM config
    lgbm_params = config['lightgbm']