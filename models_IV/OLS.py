from sklearn.linear_model import LinearRegression
from models_IV.base_model import BaseModel
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from scipy import stats

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
    
    def _predict(self, test_data, feature_cols):
        if not feature_cols:
            raise ValueError("feature_cols list cannot be empty.")
        
        X = test_data[feature_cols]
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_hat):
        """
        Evaluates the model using the provided features and target variable.   
        """ 
        metrics = {
        'MAE': mean_absolute_error(y_true, y_hat),
        'MSE': mean_squared_error(y_true, y_hat),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_hat)),
        'R2': r2_score(y_true, y_hat)
        }
        print(metrics)
        return metrics
    
    def regression_summary(self, model, X, y, feature_names=None):
        """
        Generate a summary table for a fitted sklearn LinearRegression model.
        
        Parameters:
            model: Fitted sklearn.linear_model.LinearRegression
            X: 2D numpy array or DataFrame (features used to fit the model)
            y: 1D array-like (target)
            feature_names: list of feature names (optional)
        
        Returns:
            A pandas DataFrame containing regression summary.
        """
        # Ensure input is array
        X = np.asarray(X)
        y = np.asarray(y)
        n, k = X.shape

        # Add intercept manually if fit_intercept=True
        if model.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
            feature_names = ['Intercept'] + (feature_names if feature_names is not None else [f'x{i}' for i in range(k)])
        else:
            X_design = X
            feature_names = feature_names if feature_names is not None else [f'x{i}' for i in range(k)]

        # Predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals ** 2)
        MSE = RSS / (n - X_design.shape[1])

        # Standard errors
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(MSE * XtX_inv))

        # Coefficients
        if model.fit_intercept:
            coef = np.append(model.intercept_, model.coef_)
        else:
            coef = model.coef_

        # t-stats and p-values
        t_stats = coef / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

        # R² and Adjusted R²
        TSS = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - RSS / TSS
        adj_R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)

        # Confidence intervals (95%)
        ci_low = coef - stats.t.ppf(0.975, df=n - k - 1) * se
        ci_high = coef + stats.t.ppf(0.975, df=n - k - 1) * se

        # Summary DataFrame
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef,
            'Std.Err': se,
            't': t_stats,
            'P>|t|': p_values,
            '[0.025': ci_low,
            '0.975]': ci_high
        })

        print(f"\nR²: {R2:.4f}, Adjusted R²: {adj_R2:.4f}, Observations: {n}")
        return summary_df



