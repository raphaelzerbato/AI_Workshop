from sklearn.linear_model import LinearRegression
from models_IV.base_model import BaseModel
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from scipy import stats

class LinearModel(BaseModel):
    def __init__(self, target_col: str, feature_cols: list, test_size: float = 0.2, random_state: int = 42, **kwargs):
        super().__init__(target_col, feature_cols, test_size, random_state, **kwargs)
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
    
    def compute_covariance_matrix(self, X_train, y_train):
        """
        Computes the covariance matrix of the model parameters.

        Parameters:
        X (pd.DataFrame): Features used to fit the model.
        y (pd.Series): Target variable used to fit the model.

        Returns:
        cov_matrix (pd.DataFrame): Covariance matrix of the model parameters.
        """

        # Add intercept manually if the model includes it
        if self.model.fit_intercept:
            X = np.column_stack((np.ones(X_train.shape[0]), X_train))
            feature_names = ['Intercept'] + list(getattr(self.model, 'feature_names_in_', [f"x{i}" for i in range(X.shape[1])]))
        else:
            X = np.array(X_train)
            feature_names =  + list(getattr(self.model, 'feature_names_in_', [f"x{i}" for i in range(X.shape[1])]))

        n, k = X.shape
        y_pred = self.model.predict(X_train)
        residuals = y_train - y_pred
        RSS = np.sum(residuals ** 2)
        
        # Compute covariance matrix
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_matrix = (RSS / (n - k)) * XtX_inv
        self.varcovar_mat = pd.DataFrame(
            cov_matrix,
            index=feature_names, 
            columns=feature_names
            )
        
        return self.varcovar_mat
        
    def compute_pvalues(self, X_train, y_train):
        """
        Compute p-values for coefficients in a sklearn LinearRegression model.

        Parameters:
            X (pd.DataFrame or np.ndarray): Feature matrix
            y (pd.Series or np.ndarray): Target vector
            model (LinearRegression): A fitted sklearn model

        Returns:
            pd.Series: p-values for each coefficient
        """
        # Add intercept manually if the model includes it
        if self.model.fit_intercept:
            X = np.column_stack((np.ones(X_train.shape[0]), X_train))
            feature_names = ['Intercept'] + list(getattr(self.model, 'feature_names_in_', [f"x{i}" for i in range(X.shape[1])]))
        else:
            X = np.array(X_train)
            feature_names =  list(getattr(self.model, 'feature_names_in_', [f"x{i}" for i in range(X.shape[1])]))

        # Predictions and residuals
        y_pred = self.model.predict(X_train)
        residuals = y_train - y_pred

        # Degrees of freedom
        n = X.shape[0]
        k = X.shape[1]
        dof = n - k

        # Estimate variance of residuals
        residual_var = np.sum(residuals**2) / dof

        # Variance-covariance matrix
        XTX_inv = np.linalg.inv(X.T @ X)
        var_b = residual_var * XTX_inv

        # Standard errors
        se_b = np.sqrt(np.diag(var_b))

        # t-statistics
        t_stats = self.model.coef_ if not self.model.fit_intercept else np.insert(self.model.coef_, 0, self.model.intercept_)
        t_stats = t_stats / se_b

        # Two-tailed p-values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
        self.pvalues =pd.Series(p_values, index=feature_names)
        
        return self
        
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
    



