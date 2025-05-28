from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def preprocess(self, X):
        """
        Abstract method to preprocess input data.

        Parameters:
            X (pandas.DataFrame): Input features to preprocess.

        Returns:
            Preprocessed data (type depends on implementation).
        """
        pass
    
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
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
