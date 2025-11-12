import mlflow
import pandas as pd

run_id = "3375cbdc11204be389c3563a99ccbff6"
# run_id="bcc784762a734f0c8ce1afcd5a404ce6"


model = mlflow.pyfunc.load_model(f"runs:/{run_id}/xgboost")

data = {
    "fixed acidity": [1.245, -0.582, 0.934],
    "volatile acidity": [0.478, -1.203, 0.652],
    "citric acid": [1.112, -0.356, 0.894],
    "residual sugar": [-0.452, 0.738, -0.129],
    "chlorides": [0.265, -0.874, 0.502],
    "free sulfur dioxide": [0.391, -1.026, 0.745],
    "total sulfur dioxide": [-0.214, 0.882, -0.633],
    "density": [0.672, -0.541, 1.245],
    "pH": [1.038, -0.692, 0.325],
    "sulphates": [-0.781, 0.945, -0.356],
    "alcohol": [0.582, 1.874, -0.124]
}



df = pd.DataFrame(data)

predictions = model.predict(df)
print("Predictions:", predictions)
