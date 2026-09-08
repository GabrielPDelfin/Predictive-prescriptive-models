import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. GENERATE SAMPLE HISTORICAL DATA
# ============================================================

np.random.seed(42)

n_weeks = 104  # 2 years of weekly data

data = pd.DataFrame({
    "week": np.arange(1, n_weeks + 1)
})

# Calendar features
data["week_of_year"] = ((data["week"] - 1) % 52) + 1
data["month"] = ((data["week_of_year"] - 1) // 4) + 1

# Other explanatory variables
data["price"] = np.random.normal(10, 0.5, n_weeks)

data["promotion"] = np.random.binomial(
    1,
    0.25,
    n_weeks
)

data["holiday"] = np.random.binomial(
    1,
    0.10,
    n_weeks
)

data["temperature"] = np.random.normal(
    20,
    5,
    n_weeks
)


# ============================================================
# 2. GENERATE DEMAND
# ============================================================

seasonality = (
    10 * np.sin(
        2 * np.pi * data["week_of_year"] / 52
    )
)

noise = np.random.normal(
    0,
    8,
    n_weeks
)

data["demand"] = (
    120
    + seasonality
    - 4 * data["price"]
    + 25 * data["promotion"]
    + 20 * data["holiday"]
    + 0.8 * data["temperature"]
    + noise
)


# Previous week's demand
data["previous_week_demand"] = (
    data["demand"].shift(1)
)

# Remove first observation
data = data.dropna().reset_index(drop=True)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

features = data.columns[data.columns != 'demand']

X = data[features]
y = data["demand"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

# Since this is time-series data, we use the past
# for training and the future for testing.

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))


# ============================================================
# 5. TRAIN PREDICTIVE MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\nMODEL PERFORMANCE")
print("------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("------------------")
print(importance)


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

model_data = {
    "model": model,
    "features": features
}

joblib.dump(
    model_data,
    "demand_model.pkl"
)

print("\nModel saved as demand_model.pkl")
