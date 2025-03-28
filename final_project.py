import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
df = pd.read_csv("final_project_dataset.csv")
print(df.head())
print(df.columns)

# Selecting relevant columns
columns_to_keep = ['Service Type', 'Creation Date', 'Vehicle Sales Date', 'Vehicle Model',
                   'Mileage', 'Engine Hours', 'Bill to', 'Part Amount',
                   'Labour Amount', 'Concern Code']
df = df[columns_to_keep]  # Selecting only necessary columns
print(df.info())

# Checking missing values
print(df.isnull().sum())

# Handling missing values
nan_col = ['Service Type', 'Vehicle Sales Date', 'Engine Hours', 'Concern Code']
df.dropna(subset=nan_col, inplace=True)  # Fixing inplace assignment issue

# Identifying categorical and numerical columns
filter_cat_col = df.select_dtypes(include="object")
filter_num_col = df.select_dtypes(exclude="object")

print("Categorical columns:", filter_cat_col.columns)
print("Numerical columns:", filter_num_col.columns)

# Define categorical and numerical columns explicitly
cat_col = ['Service Type', 'Vehicle Model', 'Bill to', 'Concern Code']
num_col = ['Mileage', 'Engine Hours', 'Part Amount', 'Labour Amount']

# Convert date columns to datetime
date_columns = ["Creation Date", "Vehicle Sales Date"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print(df[date_columns].dtypes)

# Extract year and month from "Creation Date"
df["year"] = df["Creation Date"].dt.year
df["month"] = df["Creation Date"].dt.month

# Standardizing numerical features
scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])

# Encoding categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[cat_col])
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_col))

# Concatenating encoded categorical features with numerical features
df_processed = pd.concat([df[num_col], encoded_df], axis=1)
print(df_processed.columns)

# Drop categorical columns after encoding
df.drop(columns=cat_col, inplace=True)

print("Remaining columns after encoding:", df.columns)

# Set plot style
sns.set_style("whitegrid")

# Plot distribution of numerical columns
plt.figure(figsize=(10, 6))
for i, col in enumerate(num_col, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

num_cols = ['Mileage', 'Engine Hours', 'Part Amount', 'Labour Amount']

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df_processed[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

Q1 = df_processed[num_cols].quantile(0.25)
Q3 = df_processed[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Define outlier limits
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = ((df_processed[num_cols] < lower_bound) | (df_processed[num_cols] > upper_bound))
print("Number of outliers per column:\n", outliers.sum())

df_processed = df_processed[~outliers.any(axis=1)]

for col in num_cols:
    lower = df_processed[col].quantile(0.05)  # 5th percentile
    upper = df_processed[col].quantile(0.95)  # 95th percentile
    df_processed[col] = np.clip(df_processed[col], lower, upper)

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df_processed[col])
    plt.title(f"Boxplot of {col} (After Outlier Handling)")

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# Define features (X) and target variables (y)
X = df_processed.drop(columns=["Part Amount", "Labour Amount"])  # Features
y = df_processed[["Part Amount", "Labour Amount"]]  # Targets

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train-Test Split Completed:")
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit on training data & transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed!")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

print("Random Forest Model Trained!")

# Predict on test set
y_pred = rf_model.predict(X_test_scaled)

# Convert to DataFrame for better readability
y_pred_df = pd.DataFrame(y_pred, columns=["Predicted Part Amount", "Predicted Labour Amount"])

print(y_pred_df.head())  # View some predictions

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance scores
feature_importance = rf_model.feature_importances_
feature_names = X.columns

# Convert to DataFrame for visualization
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance in Random Forest Model")
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],         # Number of trees
    'max_depth': [10, 20, 30, None],             # Tree depth
    'min_samples_split': [2, 5, 10],             # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],               # Min samples at a leaf node
    'max_features': ['sqrt', 'log2'],            # Max features to consider
}

# Initialize the model
rf_model = RandomForestRegressor(random_state=42)

# Use RandomizedSearchCV to find best parameters
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=10,  # Number of different combinations to test
    cv=5,       # 5-fold cross-validation
    scoring='neg_mean_absolute_error',  # Minimize MAE
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# Fit on training data
random_search.fit(X_train_scaled, y_train)

# Get best parameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)


# Train model with best parameters
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train_scaled, y_train)

print("Optimized Random Forest Model Trained!")

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predict on test set
y_pred_best = best_rf.predict(X_test_scaled)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred_best)
mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)

print(f"Optimized Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
