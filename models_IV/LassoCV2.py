import numpy as np
import pandas as pd
from models_IV.base_model import BaseModel
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class LassoCVModel(BaseModel):
    def __init__(
            self, target_col: str, feature_cols: list,
            test_size: float = 0.2, random_state: int = 42,
            unpenalized_cols: list = None, **lasso_kwargs):
        
        super().__init__(target_col, feature_cols, test_size, random_state)

        self.unpenalized_cols = unpenalized_cols or []
        self.penalized_cols = [col for col in feature_cols if col not in self.unpenalized_cols]

        self.scaler_pen = StandardScaler()
        self.scaler_unpen = StandardScaler()

        self.lasso = LassoCV(random_state=random_state, **lasso_kwargs)
        self.ols = LinearRegression()
        self.best_alpha = None
        self.coefs = None

    def preprocess(self, X):
        # Only preprocess penalized data
        X_pen = self.scaler_pen.transform(X[self.penalized_cols])
        X_unpen = self.scaler_unpen.transform(X[self.unpenalized_cols])
        return pd.DataFrame(np.hstack([X_unpen, X_pen]), index=X.index, columns=self.unpenalized_cols + self.penalized_cols)

    def train(self, X, y):
        X_pen = self.scaler_pen.fit_transform(X[self.penalized_cols])
        X_unpen = self.scaler_unpen.fit_transform(X[self.unpenalized_cols])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("once", ConvergenceWarning)
            self.lasso.fit(X_pen, y)

            if any(issubclass(wi.category, ConvergenceWarning) for wi in w):
                print("\n⚠️ Warning: Lasso did not converge for some alpha values\n")

        residuals = y - self.lasso.predict(X_pen)
        self.ols.fit(X_unpen, residuals)

        self.best_alpha = self.lasso.alpha_
        print(f"Best Lasso regularization parameter: {self.best_alpha}")

        # Store combined coefficients
        coef_data = [self.ols.intercept_ + self.lasso.intercept_]
        coef_data += list(self.ols.coef_) + list(self.lasso.coef_)
        self.coefs = pd.DataFrame(
            np.array(coef_data).reshape((1 + len(self.unpenalized_cols) + len(self.penalized_cols), 1)),
            columns=['Coefs'],
            index=['(Intercept)'] + self.unpenalized_cols + self.penalized_cols
        )

    def _predict(self, X):
        X_pen = self.scaler_pen.transform(X[self.penalized_cols])
        X_unpen = self.scaler_unpen.transform(X[self.unpenalized_cols])

        return self.lasso.predict(X_pen) + self.ols.predict(X_unpen)

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_pred = self._predict(X_train)
        y_test_pred = self._predict(X_test)

        metrics = {
            "Train MSE": mean_squared_error(y_train, y_train_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred),
            "Train R^2": r2_score(y_train, y_train_pred),
            "Test R^2": r2_score(y_test, y_test_pred)
        }

        return metrics

    def plot_cv_path(self):
        """
        Plot cross-validated MSE vs log2(alpha) for the penalized part of the model.
        """
        mean_mse = self.lasso.mse_path_.mean(axis=1)
        std_mse = self.lasso.mse_path_.std(axis=1)
        log_alphas = np.log2(self.lasso.alphas_)

        plt.figure(figsize=(8, 5))
        plt.plot(log_alphas, mean_mse, label='Mean CV MSE', marker='o')
        plt.fill_between(log_alphas, mean_mse - std_mse, mean_mse + std_mse,
                        color='lightgray', label='±1 std dev')
        plt.axvline(np.log2(self.best_alpha), linestyle='--', color='red',
                    label=f'Best alpha = {self.best_alpha:.5f}')
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
