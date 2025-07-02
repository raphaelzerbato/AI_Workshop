from xgboost import XGBRegressor
from models_IV.base_model import BaseModel  # adjust if your path is different

class XGBoostModel(BaseModel):
    def __init__(self, target_col: str, feature_cols: list, test_size: float = 0.2, 
                 random_state: int = 42, **xgb_kwargs):

        super().__init__(target_col, feature_cols, test_size, random_state)
        self.model = XGBRegressor(
            random_state=random_state, 
            **xgb_kwargs
        )

    def train(self, X, y):
        """
        Fit the XGBoost regressor on training data.
        """
        self.model.fit(X, y)

    def _predict(self, X):
        """
        Make predictions using the fitted XGBoost regressor.
        """
        return self.model.predict(X)
