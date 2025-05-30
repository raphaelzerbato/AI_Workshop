from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PCAWrapper:
    def __init__(self, variance_ratio_preserved: float = 1.0, feature_cols: list = None, test_size:float = 0.2):
        if not (0 < variance_ratio_preserved <= 1):
            raise ValueError("variance_ratio_preserved must be between 0 and 1.")
        
        self.variance_ratio_preserved = variance_ratio_preserved
        self.scaler = StandardScaler()
        self.pca = None
        self.nb_selected_components = None
        self.test_size = test_size
        self.feature_cols = feature_cols if feature_cols is not None else []
    
    def _train_test_split(self, df, **kwargs):
        X = df[self.feature_cols]
        self.X_train, self.X_test = train_test_split(
            X, test_size=self.test_size, **kwargs
        )

    def fit(self, X: pd.DataFrame):
        """
        Fit scaler and PCA model on training data.
        """
        X_scaled = self.scaler.fit_transform(X)

        # Apply full PCA (we'll later select components based on explained variance)
        self.full_pca = PCA(n_components=X.shape[1])
        self.full_pca.fit(X_scaled)

        # Compute how many components to keep
        cum_var_ratio = self.full_pca.explained_variance_ratio_.cumsum()
        self.nb_selected_components = np.searchsorted(cum_var_ratio, self.variance_ratio_preserved) + 1
        print(f"\n✅ The first {self.nb_selected_components} components preserve at least {self.variance_ratio_preserved:.2f} of the variance.\n")

        # Now refit PCA with selected number of components
        self.pca = PCA(n_components=self.nb_selected_components)
        self.pca.fit(X_scaled)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data into PCA space using fitted scaler and PCA model.
        """
        if self.pca is None:
            raise RuntimeError("You must call `fit()` before `transform()`.")

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return pd.DataFrame(
            X_pca,
            index=X.index,
            columns=[f"PC_{i+1}" for i in range(self.nb_selected_components)]
        )

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Fit on X_train and transform both X_train and X_test.
        """
        self.fit(X_train)
        X_train_pca = self.transform(X_train)
        X_test_pca = self.transform(X_test)
        return X_train_pca, X_test_pca

    def plot_variance_explained(self):
        """
        Plot explained variance ratio and cumulative variance ratio.
        """
        if self.pca is None:
            raise RuntimeError("PCA has not been fitted yet. Call `fit()` first.")

        x = np.arange(len(self.full_pca.explained_variance_ratio_))
        y = self.full_pca.explained_variance_ratio_

        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=x, y=y, ax=ax1, color='blue', label='Explained variance ratio')
        ax1.set_xlabel("Component index")
        ax1.set_ylabel("Explained variance ratio", color='blue')

        ax2 = ax1.twinx()
        sns.lineplot(x=x, y=y.cumsum(), ax=ax2, color='orange', linestyle='--', label='Cumulative variance ratio')
        ax2.set_ylabel("Cumulative variance ratio", color='orange')

        ax1.axvline(self.nb_selected_components - 1, color='red', linestyle='--', label=f'{self.nb_selected_components} components')
        ax1.legend(loc='center right')
        plt.title("PCA: Explained and Cumulative Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
