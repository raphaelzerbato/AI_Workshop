import numpy as np
import pandas as pd
from models_IV.base_model import BaseModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class LassoCVModel(BaseModel):
    def __init__(
            self, target_col: str, feature_cols: list, test_size: float = 0.2,
            random_state: int = 42, **lasso_kwargs):
        super().__init__(target_col, feature_cols, test_size, random_state)
        self.scaler = StandardScaler()
        self.model = LassoCV(
            **lasso_kwargs  
        )
        self.best_alpha = None
        self.coefs = None

    def preprocess(self, X):
        return pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("once", ConvergenceWarning)
            self.model.fit(X_scaled, y)

            if any(issubclass(wi.category, ConvergenceWarning) for wi in w):
                print("\n⚠️ Warning: Lasso did not converge for some alpha values\n")

        self.best_alpha = self.model.alpha_
        print(f"Best Lasso regularization parameter: {self.best_alpha}")

        # Store coefficients
        coefs = [self.model.intercept_] + list(self.model.coef_)
        self.coefs = pd.DataFrame(
            np.array(coefs).reshape((X.shape[1] + 1, 1)),
            columns=['Coefs'],
            index=['(Intercept)'] + list(X.columns)
        )

    def _predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_pred = self._predict(X_train)
        y_test_pred = self._predict(X_test)

        metrics = {
            "Train RMSE": mean_squared_error(y_train, y_train_pred),
            "Test RMSE": mean_squared_error(y_test, y_test_pred),
            "Train R^2": r2_score(y_train, y_train_pred),
            "Test R^2": r2_score(y_test, y_test_pred)
        }

        return metrics

    def plot_cv_path(self):
        # 6. Plot CV MSE vs log(alpha)
        mean_mse = self.model.mse_path_.mean(axis=1)
        std_mse = self.model.mse_path_.std(axis=1)
        log_alphas = np.log2(self.model.alphas_)
        
        plt.plot(log_alphas, mean_mse, label='Mean CV MSE', marker='o')
        plt.fill_between(log_alphas, mean_mse - std_mse, mean_mse + std_mse, color='lightgray', label='±1 std dev')
        plt.axvline(np.log2(self.model.best_alpha), linestyle='--', color='red', label=f'Best lambda = {self.model.best_alpha:.5f}')
        plt.xlabel('Log2(Alpha)')
        plt.xlabel('Log2(Alpha)')
        plt.ylabel('Mean Squared Error (CV)')
        plt.title('LassoCV: CV Error vs log2(Regularization Strength)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_true_predicted(self, Y_true, Y_pred, VarName=""):
        """
        Show scatter plot of two pd.Series: Y_true in x-axis, Y_pred in y-axis
        """
        r2 = r2_score(Y_true, Y_pred)

        sns.scatterplot(x=Y_true, y=Y_pred)
        plt.plot(Y_true, Y_true, linestyle='--', color='gray')
        plt.xlabel(f"True {VarName}")
        plt.ylabel(f"Predicted {VarName}")
        plt.title(f"Prediction with CV Lasso\n(R² = {round(r2, 4)})")
        plt.show()
