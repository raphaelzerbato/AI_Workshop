from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.y_train_pred = self._predict(X_train)
        self.y_test_pred = self._predict(X_test)

        metrics = {
            "Train MSE": mean_squared_error(y_train, self.y_train_pred),
            "Test MSE": mean_squared_error(y_test, self.y_test_pred),
            "Train R^2": r2_score(y_train, self.y_train_pred),
            "Test R^2": r2_score(y_test, self.y_test_pred)
        }
        return metrics

    def plot_true_predicted(self, Y_true, Y_pred, VarName=""):
        """
        Scatter plot of true vs predicted values
        """
        r2 = r2_score(Y_true, Y_pred)
        sns.scatterplot(x=Y_true, y=Y_pred)
        plt.plot(Y_true, Y_true, linestyle='--', color='gray')
        plt.xlabel(f"True {VarName}")
        plt.ylabel(f"Predicted {VarName}")
        plt.title(f"Random Forest Predictions\n(R² = {round(r2, 4)})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
