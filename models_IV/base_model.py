from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel(ABC):
     
    def __init__(
            self,
            target_col: str,
            feature_cols: list, 
            test_size: float = 0.2, 
            random_state: int = 42,
            **extra_init_args
        ):
        
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state
        self.extra_init_args = extra_init_args
    
    def _train_test_split(self, df, **kwargs):
        X = df[self.feature_cols]
        y = df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, **kwargs
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def copy(self):
        """
        Creates a shallow copy of the model instance with the same initialization parameters.

        Returns:
            A new instance of the same class with copied initialization parameters.
        """
        return self.__class__(
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            test_size=self.test_size,
            random_state=self.random_state,
            **self.extra_init_args
        )

    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def _predict(self, X):
        pass

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
    
    def plot_true_predicted(self, Y_true, Y_pred, ModelName = "", VarName=""):
        """
        Scatter plot of true vs predicted values
        """
        r2 = r2_score(Y_true, Y_pred)
        sns.scatterplot(x=Y_true, y=Y_pred)
        plt.plot(Y_true, Y_true, linestyle='--', color='gray')
        plt.xlabel(f"True {VarName}")
        plt.ylabel(f"Predicted {VarName}")
        plt.title(f"{ModelName} Predictions\n(R² = {round(r2, 4)})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
