from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd

class BaseModel(ABC):
     
    def __init__(self, target_col: str, feature_cols: list, test_size: float = 0.2, random_state: int = 42):
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state
    
    def _train_test_split(self, df, **kwargs):
        X = df[self.feature_cols]
        y = df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, **kwargs
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Abstract method to evaluate the model.

        Parameters:
            X (pandas.DataFrame): Input features for evaluation.
            y (pandas.Series): True labels for evaluation.

        Returns:
            Evaluation metrics (type depends on implementation).
        """
        pass
