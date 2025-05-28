from sklearn.linear_model import LinearRegression
from models_IV.base_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self,**kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, train_data, target_col, feature_cols, **kwargs):
        """
        Fits the linear model to the provided data.

        Parameters:
        data (pd.DataFrame): The input data containing features and target.
        target_col (str): The name of the target column.
        feature_cols (list): List of feature column names.

        Returns:
        self: Returns the instance itself.
        """
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        if target_col not in train_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        for col in feature_cols:
            if col not in train_data.columns:
                raise ValueError(f"Feature column '{col}' not found in data.")

        X = train_data[feature_cols   ]   
        y = train_data[target_col]
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, test_data, feature_cols):
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        
        X = test_data[feature_cols]
        return self.model.predict(X)

