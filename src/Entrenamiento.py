import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn import metrics

df = pd.read_csv("../data/train/coches_train.csv")

# x = df[["kms", "power", "antiquity", "doors", "num_make"]]
x = df[["kms", "power", "antiquity", "doors"]]
#x = df.drop(columns=["make", "model", "version", "fuel", "shift", "color"])
y = df["price"]

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=10)

# Modelo Lineal Regresion.
modelo_lr = LinearRegression()
modelo_lr.fit(x_train, y_train)

predictions_lr = modelo_lr.predict(x_test)

print("MAE:", metrics.mean_absolute_error(y_test, predictions_lr))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, predictions_lr))
print("MSE:", metrics.mean_squared_error(y_test, predictions_lr))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions_lr)))
print("r2_score train", modelo_lr.score(x_train, y_train))
print("r2_score test",modelo_lr.score(x_test, y_test))

with open('../models/modelo_LinearRegression/modelo_LinearRegression.pkl', "wb") as archivo:
    pickle.dump(modelo_lr, archivo)

# Modelo Lineal Regresion con regresión polinomial de 3.
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

poly_feats = PolynomialFeatures(degree = 3)
poly_feats.fit(x_scaler)
x_poly = poly_feats.transform(x_scaler)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_poly,y, test_size = 0.2, random_state=12)

modelo_reg_pol = LinearRegression()
modelo_reg_pol.fit(x_train_2, y_train_2)

predictions_reg_pol = modelo_reg_pol.predict(x_test_2)

print("MAE:", metrics.mean_absolute_error(y_test_2, predictions_reg_pol))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test_2, predictions_reg_pol))
print("MSE:", metrics.mean_squared_error(y_test_2, predictions_reg_pol))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test_2, predictions_reg_pol)))
print("r2_score train", modelo_reg_pol.score(x_train_2, y_train_2))
print("r2_score test",modelo_reg_pol.score(x_test_2, y_test_2))

with open('../models/modelo_Polinomial_regression/modelo_Polinomial_regression.pkl', "wb") as archivo:
    pickle.dump(modelo_reg_pol, archivo)

# Modelo DecisionTreeRegressor.
modelo_dec_tree = DecisionTreeRegressor(max_depth=20, random_state=42)
modelo_dec_tree.fit(x_train, y_train)

predictions_dec_tree = modelo_dec_tree.predict(x_test)

print("MAE:", metrics.mean_absolute_error(y_test, predictions_dec_tree))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, predictions_dec_tree))
print("MSE:", metrics.mean_squared_error(y_test, predictions_dec_tree))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions_dec_tree)))
print("r2_score train", modelo_dec_tree.score(x_train, y_train))
print("r2_score test",modelo_dec_tree.score(x_test, y_test))

with open('../models/modelo_DecisionTreeRegressor/modelo_DecisionTreeRegressor.pkl', "wb") as archivo:
    pickle.dump(modelo_dec_tree, archivo)

# Modelo Random Forest.
modelo_rf = RandomForestRegressor(n_estimators=200, random_state=42)
modelo_rf.fit(x_train, y_train)

mae_scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False)

scores = cross_val_score(modelo_rf, x_test, y_test, cv=5, scoring=mae_scorer)

print("Resultados de la validación cruzada:")
print("Scores (negativo MAE):", scores)
print("Promedio del MAE:", -scores.mean())

predictions_rf = modelo_rf.predict(x_test)

print("MAE:", metrics.mean_absolute_error(y_test, predictions_rf))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, predictions_rf))
print("MSE:", metrics.mean_squared_error(y_test, predictions_rf))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions_rf)))
print("r2_score train", modelo_rf.score(x_train, y_train))
print("r2_score test",modelo_rf.score(x_test, y_test))

with open('../models/modelo_RandomForestRegressor/modelo_RandomForestRegressor.pkl', "wb") as archivo:
    pickle.dump(modelo_rf, archivo)

# Modelo KNeighborsRegressor.
knn_reg = KNeighborsRegressor(n_neighbors=3)

knn_reg.fit(x_train, y_train)

predictions_knn = knn_reg.predict(x_test)

print("MAE:", metrics.mean_absolute_error(y_test, predictions_knn))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, predictions_knn))
print("MSE:", metrics.mean_squared_error(y_test, predictions_knn))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions_knn)))
print("r2_score train", knn_reg.score(x_train, y_train))
print("r2_score test",knn_reg.score(x_test, y_test))

with open('../models/modelo_KNeighborsRegressor/modelo_KNeighborsRegressor.pkl', "wb") as archivo:
    pickle.dump(knn_reg, archivo)

# Modelo Pipeline(PCA y KNeighborsRegressor).

modelo_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('knn', KNeighborsRegressor(n_neighbors=3))
])

modelo_pipeline.fit(x_train, y_train)

prediciones_pipeline = modelo_pipeline.predict(x_test)

print("MAE:", metrics.mean_absolute_error(y_test, prediciones_pipeline))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, prediciones_pipeline))
print("MSE:", metrics.mean_squared_error(y_test, prediciones_pipeline))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, prediciones_pipeline)))
print("r2_score train", modelo_pipeline.score(x_train, y_train))
print("r2_score test",modelo_pipeline.score(x_test, y_test))

with open('../models/modelo_Pipeline/modelo_Pipeline.pkl', "wb") as archivo:
    pickle.dump(modelo_pipeline, archivo)