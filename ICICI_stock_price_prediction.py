from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
file_path = "icici_dataset.csv"
df = pd.read_csv(file_path)
df.info()

# Trim column names and remove spaces
df.columns = df.columns.str.strip()

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


# Extract date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# Clean numeric columns and convert to float
num_cols = ["OPEN", "HIGH", "LOW", "PREV. CLOSE", "ltp", "close", "vwap", "VOLUME", "VALUE"]

for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

print(df.info())

print(df.isna().values.any())

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Date", "OPEN", "HIGH", "LOW", "PREV. CLOSE", "close", "VOLUME", "VALUE"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Set plot style
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# Create a figure and a 3x1 grid of subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# First subplot for OPEN and CLOSE prices
axs[0].plot(df['Date'], df["OPEN"], label='OPEN')
axs[0].plot(df['Date'], df["close"], label='CLOSE')
axs[0].set_title('Stock Opening and Closing Prices')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price in Rupees')
axs[0].legend()

# Second subplot for HIGH and LOW prices
axs[1].plot(df['Date'], df["HIGH"], label='HIGH')
axs[1].plot(df['Date'], df["LOW"], label='LOW')
axs[1].set_title('Stock High and Low Prices')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Price in Rupees')
axs[1].legend()

# Third subplot for VOLUME and VALUE
axs[2].plot(df['Date'], df["VOLUME"], label='VOLUME')
axs[2].plot(df['Date'], df["VALUE"], label='VALUE')
axs[2].set_title('Stock Volume and Value')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Value')
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plots
plt.show()

# Select features and target
features = ["Year", "Month", "Day", "HIGH", "LOW", "VOLUME"]
target = "close"

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")


# --- ✅ CORRECTED NEW DATA FOR PREDICTION ---
new_data = {"date": "2025-04-11", "high": 960, "low": 948, "volume": 1200000}

# Convert date to features
date_obj = pd.to_datetime(new_data["date"])
year, month, day = date_obj.year, date_obj.month, date_obj.day

# Prepare input in the correct format
input_data = np.array([[year, month, day, new_data["high"], new_data["low"], new_data["volume"]]])

# Standardize new data
input_scaled = scaler.transform(input_data)

# Make prediction
predicted_price = model.predict(input_scaled)[0]

print(f"Predicted Closing Price on {new_data['date']}: ₹{round(predicted_price, 2)}")