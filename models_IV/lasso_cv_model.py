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
            self, target_col: str, feature_cols: list, test_size: float = 0.2,
            random_state: int = 42, **lasso_kwargs):
        super().__init__(target_col, feature_cols, test_size, random_state)
        self.scaler = StandardScaler()
        self.model = LassoCV(
            random_state=random_state, **lasso_kwargs  
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
            "Train MSE": mean_squared_error(y_train, y_train_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred),
            "Train R^2": r2_score(y_train, y_train_pred),
            "Test R^2": r2_score(y_test, y_test_pred)
        }

        return metrics

    def plot_cv_path(self):
        # Plot CV MSE vs log(alpha)
        mean_mse = self.model.mse_path_.mean(axis=1)
        std_mse = self.model.mse_path_.std(axis=1)
        log_alphas = np.log2(self.model.alphas_)

        plt.plot(log_alphas, mean_mse, label='Mean CV MSE', marker='o')
        plt.fill_between(log_alphas, mean_mse - std_mse, mean_mse + std_mse, color='lightgray', label='±1 std dev')
        plt.axvline(np.log2(self.best_alpha), linestyle='--', color='red', label=f'Best lambda = {self.best_alpha:.5f}')
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



class LassoCVModel2(LassoCVModel):

    """
    LassoCVModel2 extends LassoCVModel to handle both penalized and unpenalized features.
    It uses LassoCV for penalized features and OLS for unpenalized features, in a FWL two-step process:
    1. Fit the target on penalized features using LassoCV.
    2. Compute residuals from LassoCV and fit these residuals on unpenalized features using OLS.
    """

    def __init__(
            self, target_col: str, feature_cols: list,
            test_size: float = 0.2, random_state: int = 42,
            unpenalized_cols: list = None, **lasso_kwargs):

        super().__init__(target_col, feature_cols, test_size, random_state, **lasso_kwargs)

        self.unpenalized_cols = unpenalized_cols or []
        self.penalized_cols = [col for col in feature_cols if col not in self.unpenalized_cols]

        # Override scalers and add and OLS model
        self.scaler_pen = StandardScaler()
        self.scaler_unpen = StandardScaler()
        self.ols = LinearRegression(fit_intercept=False)

    def preprocess(self, X):
        # Preprocess unpenalized and penalized features separately
        X_unpen = self.scaler_unpen.transform(X[self.unpenalized_cols])
        X_pen = self.scaler_pen.transform(X[self.penalized_cols])
        return pd.DataFrame(
            np.hstack([X_unpen, X_pen]),
            index=X.index,
            columns=self.unpenalized_cols + self.penalized_cols
        )

    def train(self, X, y):
        X_pen = pd.DataFrame(
            self.scaler_pen.fit_transform(X[self.penalized_cols]),
            columns=self.penalized_cols, index=X.index
            )
        X_unpen = self.scaler_unpen.fit_transform(X[self.unpenalized_cols])

        self.ols.fit(X_unpen, y)
        residuals = y - self.ols.predict(X_unpen)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("once", ConvergenceWarning)
            self.model.fit(X_pen, residuals)
            if any(issubclass(wi.category, ConvergenceWarning) for wi in w):
                print("\n⚠️ Warning: Lasso did not converge for some alpha values\n")

        self.best_alpha = self.model.alpha_
        print(f"Best Lasso regularization parameter: {self.best_alpha}")

        # Get selected features from penalized ones, i.e. penalized_cols with non-zero coefficients
        self.selected_penalized_cols = [col for col, coef in zip(self.penalized_cols, self.model.coef_) if coef != 0]
        # Now fit an ols model that will give the right coefficients for the unpenalized features
        X_sel_pen = X_pen[self.selected_penalized_cols]
        ols_with_intercept = LinearRegression(fit_intercept=True)
        ols_with_intercept.fit(X_sel_pen, y)
        residuals = y - ols_with_intercept.predict(X_sel_pen)
        self.ols.fit(X_unpen, residuals)

        # Combine coefficients
        coef_data = [self.model.intercept_]
        coef_data += list(self.ols.coef_) + list(self.model.coef_)
        self.coefs = pd.DataFrame(
            np.array(coef_data).reshape((1 + len(self.unpenalized_cols) + len(self.penalized_cols), 1)),
            columns=['Coefs'],
            index=['(Intercept)'] + self.unpenalized_cols + self.penalized_cols
        )

    def _predict(self, X):
        X_pen = self.scaler_pen.fit_transform(X[self.penalized_cols])
        X_unpen = self.scaler_unpen.fit_transform(X[self.unpenalized_cols]) 
        return self.model.predict(X_pen) + self.ols.predict(X_unpen)    