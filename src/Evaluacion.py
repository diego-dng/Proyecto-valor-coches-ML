import numpy as np
import pandas as pd
import joblib
from sklearn import metrics

# Cargamos el modelo final.
modelo = joblib.load('../models/modelo_RandomForestRegressor/modelo_RandomForestRegressor.pkl')


# Cargamos el csv de test.
df = pd.read_csv("../data/test/coches_test.csv")

# Realizo la predicción con el modelo.
x = df[["kms", "power", "doors", "antiquity"]]

precio_pred = modelo.predict(x)

print("MAE:", metrics.mean_absolute_error(df["price"], precio_pred))
print("MAPE:", metrics.mean_absolute_percentage_error(df["price"], precio_pred))
print("MSE:", metrics.mean_squared_error(df["price"], precio_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(df["price"], precio_pred)))
print("r2_score test",modelo.score(x, precio_pred))