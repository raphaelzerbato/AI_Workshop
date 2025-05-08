# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import shap
from statsmodels.formula.api import ols
from statsmodels.api import Probit

np.random.seed(42)  # for reproducibility

# Number of observations
n = 1000

# %%
# Controls
hotel_types = np.random.choice(['Luxury', 'Standard'], size=n, p=[0.3, 0.7])
locations = np.random.choice(['Beach', 'Mountain', 'City'], size=n, p=[0.4, 0.3, 0.3])
dates = np.random.choice(pd.date_range('2024-01-01', '2024-12-31', freq='D'), size=n)

# %%
# Instrument: Local minimum wage shock (exogenous)
min_wage_shock = np.random.normal(loc=0, scale=1, size=n)

# %%
# Price per night (endogenous: affected by demand shock)
demand_shock = np.random.normal(0, 1, n)
base_price = 100 + 50 * (hotel_types == 'Luxury') + 20 * (locations == 'Beach')
price = base_price + 5 * min_wage_shock + 30 * demand_shock + np.random.normal(0, 10, n)

# %%
# True causal effect of price on occupancy is negative
# But demand shocks drive both price ↑ and occupancy ↑ (confounding!)
occupancy_rate = 0.8 - 0.0015 * price + 0.05 * demand_shock + np.random.normal(0, 0.05, n)
occupancy_rate = np.clip(occupancy_rate, 0, 1)

# %%
# Build dataframe
data = pd.DataFrame({
    'date': dates,
    'hotel_type': hotel_types,
    'location': locations,
    'min_wage_shock': min_wage_shock,
    'price': price,
    'occupancy_rate': occupancy_rate
})

print(data.head())
# %%
# Prepare the data
X = data.drop(columns=['occupancy_rate'])
X = pd.get_dummies(X, columns=['hotel_type', 'location'], drop_first=True)  # One-hot encode categorical variables
X['date'] = X['date'].apply(lambda x: x.timestamp()).astype('int64')  # Convert date to numeric format
y = data['occupancy_rate']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# %%
# Explain the model's predictions using SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Generate SHAP summary plot for the "price" variable
price_index = list(X_test.columns).index('price')
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Generate the marginal contribution plot for the "price" variable
shap.dependence_plot('price', shap_values.values, X_test, feature_names=X_test.columns)

# %% 
import statsmodels.api as sm

# Step 1: Estimate the selection equation using Probit
data['selected'] = (data['price'] > 0).astype(int)  # Selection indicator
probit_model = Probit(data['selected'], sm.add_constant(data[['min_wage_shock']]))
probit_results = probit_model.fit()
data['inverse_mills_ratio'] = probit_results.predict(sm.add_constant(data[['min_wage_shock']]), linear=True)

# Step 2: Estimate the outcome equation with the inverse Mills ratio
outcome_model = ols('occupancy_rate ~ price + inverse_mills_ratio', data=data).fit()

# Print the summary of the outcome model
print(outcome_model.summary())
