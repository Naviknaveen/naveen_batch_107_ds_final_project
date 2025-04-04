import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Step 1: Load Dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)
print("File loaded successfully")

# Display Basic Information
print(df.info())
print("Missing values per column:")
print(df.isnull().sum())
print("Dataset shape:", df.shape)

# Step 2: Data Cleaning and Preprocessing
# Handle missing values
nan_columns = ['Service Subtype', 'Vehicle Sales Date', 'License Plate', 'Point Of Contact name', 'Point of contact Mobile', 'Engine Hours']
df.dropna(subset=nan_columns, inplace=True)
print("Dropped rows with NaN values in critical columns.")

# Drop unwanted columns
columns_to_drop = ["Point Of Contact name", "Point of contact Mobile", "Service Subtype", "License Plate"]
df.drop(columns=columns_to_drop, inplace=True)
print("Dropped unwanted columns")
print(df.info())

# Convert Date Columns to Datetime Format
date_cols = ["JC Creation Date", "JC Closed Date", "Vehicle Sales Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
print("Converted Date Columns to Datetime Format")
print(df[date_cols].dtypes)

# Convert Numeric Columns
numeric_cols = ["Mileage", "Customer Amount"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
print("Converted to Numeric Columns")
print(df[numeric_cols].dtypes)

# Detect & Remove Outliers using IQR
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numeric_cols:
    df = remove_outliers(df, col)
print("Outlier Removal Completed")

# Step 3: Exploratory Data Analysis (EDA)
# Set visualization style
sns.set_style("whitegrid")

# Plot distributions of numerical features
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

sns.histplot(df["Mileage"].dropna(), bins=30, kde=True, ax=axes[0])
axes[0].set_title("Mileage Distribution")

sns.histplot(df["Engine Hours"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Engine Hours Distribution")

sns.histplot(df["Customer Amount"].dropna(), bins=30, kde=True, ax=axes[2])
axes[2].set_title("Customer Amount Distribution")

plt.show()
print("Visualized Data Distribution")

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Mileage", "Engine Hours", "Customer Amount"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Clustering (Unsupervised Learning)
features = ["Mileage", "Engine Hours", "Customer Amount"]
df_cluster = df[features].dropna()
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)
print("Standard Scaler Applied")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster["Cluster"] = kmeans.fit_predict(df_cluster_scaled)
print("Clustering Done")

# Visualizing Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_cluster["Mileage"], y=df_cluster["Customer Amount"], hue=df_cluster["Cluster"], palette="viridis")
plt.title("K-Means Clustering of Transactions")
plt.xlabel("Mileage")
plt.ylabel("Customer Amount")
plt.show()
print("Visualized Clusters")

# Supervised Learning - Random Forest Regression
# Encode categorical variables
service_encoder = LabelEncoder()
vehicle_encoder = LabelEncoder()

df["Service Type"] = service_encoder.fit_transform(df["Service Type"])
df["Vehicle Model"] = vehicle_encoder.fit_transform(df["Vehicle Model"])

# Define Features and Target Variable
X = df.drop(columns=["Customer Amount", "Job Card number", "VIN Number", "Customer Name", "Customer Invoice Number", "Dicv Amount", "JC Creation Date", "JC Closed Date", "Vehicle Sales Date"])
y = df["Customer Amount"].dropna()

print("Feature Columns:", X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Test Split Completed")

# Train Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("Random Forest Model Trained")

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.2f}")


import pickle
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)