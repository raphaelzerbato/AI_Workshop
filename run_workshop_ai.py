# %%
import data_preprocessing as dp
from data_preprocessing import *


# %%

def splitting_data(data:pd.dataframe, Y_var:str)->pd.dataframe:
    # Splitting the data into train and test sets
    X = data.drop(columns=[Y_var])
    y = data[Y_var]
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 



# %%
if __name__ == "__main__":

    # %%    
    # Load data and config
    cars_file_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/data/data_cars/car_prices.csv'
    population_file_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/data/data_cars/states_populations_yr2015.csv'

    cars_db         = dp.load_data(cars_file_path)
    population_db   = dp.load_data(population_file_path)
    
    # Load config
    config_path = 'C:/Users/rapha/PythonTutos/AI_Workshop/config/config.yaml'
    config = dp.load_config(config_path)


    # %%
    # Preprocess data
    X_train, X_test, y_train, y_test = splitting_data(cars_db, 'price')
    
    # Train model
    model_name = 'xgboost'
    trained_model = dp.train_selected_model(model_name, X_train, y_train, **config['model_params'])

