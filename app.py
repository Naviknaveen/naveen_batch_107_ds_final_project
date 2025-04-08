from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("stock_price_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input from form
            date = request.form["date"]
            high = float(request.form["high"])
            low = float(request.form["low"])
            volume = float(request.form["volume"])

            # Convert date
            date_obj = pd.to_datetime(date)
            year, month, day = date_obj.year, date_obj.month, date_obj.day

            # Prepare input for model
            features = np.array([[year, month, day, high, low, volume]])
            features_scaled = scaler.transform(features)

            # Predict closing price
            predicted_price = model.predict(features_scaled)[0]

            return render_template("index.html", prediction=predicted_price)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
