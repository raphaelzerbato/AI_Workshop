from sklearn.ensemble import RandomForestRegressor
from models_IV.base_model import BaseModel  # adjust if your import path is different

class RandomForestModel(BaseModel):
    def __init__(self, target_col: str, feature_cols: list, test_size: float = 0.2, 
                 random_state: int = 42, **rf_kwargs):
        super().__init__(target_col, feature_cols, test_size, random_state)
        self.model = RandomForestRegressor(
            random_state=random_state, **rf_kwargs 
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

