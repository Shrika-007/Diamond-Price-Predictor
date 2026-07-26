from flask import Flask, render_template, request
import joblib
import os
from xgboost import XGBRegressor
model=XGBRegressor()
model.load_model("diamond_price_model.json")
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():

    cut_map = {
        "Fair":1,
        "Good":2,
        "Very Good":3,
        "Premium":4,
        "Ideal":5
    }

    color_map = {
        "J":1,
        "I":2,
        "H":3,
        "G":4,
        "F":5,
        "E":6,
        "D":7
    }

    clarity_map = {
        "I1":1,
        "SI2":2,
        "SI1":3,
        "VS2":4,
        "VS1":5,
        "VVS2":6,
        "VVS1":7,
        "IF":8
    }

    import pandas as pd
    import numpy as np

    data = pd.DataFrame([{
        "carat": float(request.form["carat"]),
        "cut": cut_map[request.form["cut"]],
        "color": color_map[request.form["color"]],
        "clarity": clarity_map[request.form["clarity"]],
        "depth": float(request.form["depth"]),
        "table": float(request.form["table"]),
        "x": float(request.form["x"]),
        "y": float(request.form["y"]),
        "z": float(request.form["z"])
    }])

    pred = model.predict(data)[0]

    pred = np.exp(pred)

    return render_template("result.html", prediction=round(pred,2))
if __name__=="__main__":
    port = int(os.environment.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)