from sklearn.linear_model import LinearRegression
from models_IV.base_model import BaseModel
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LinearModel(BaseModel):
    def __init__(self, target_col: str, feature_cols: list, test_size: float = 0.2, random_state: int = 42, **kwargs):
        super().__init__(target_col, feature_cols, test_size, random_state)
        self.model = LinearRegression(**kwargs)

    def __preprocess(self):
        """
        Abstract method to preprocess input data.

        Parameters:
            X (pandas.DataFrame): Input features to preprocess.

        Returns:
            Preprocessed data (type depends on implementation).
        """
        pass    

    def train(self, X, y, **kwargs):
        """
        Fits the linear model to the provided data.

        Parameters:
        data (pd.DataFrame): The input data containing features and target.
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable for training.

        Returns:
        self: Returns the instance itself.
        """
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, test_data, feature_cols):
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        
        X = test_data[feature_cols]
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_hat):
        """
        Evaluates the model using the provided features and target variable.   
        """ 
        return {
        'MAE': mean_absolute_error(y_true, y_hat),
        'MSE': mean_squared_error(y_true, y_hat),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_hat)),
        'R2': r2_score(y_true, y_hat)
    }


