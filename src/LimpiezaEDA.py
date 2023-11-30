import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split

# Creo el Dataframe con los datos del csv de "coches_train.csv"
df = pd.read_csv("../data/raw/coches-de-segunda-mano-sample.csv")
df.head()

# Elimino las columnas que no necesito.
df = df.drop(columns=["url", "company", "model", "version", "color", "shift",
                       "photos", "is_professional", "price_financed",
                         "dealer", "province", "country",
                           "publish_date", "insert_date"])

# Creo la columna antiquity para saber los años de antiguedad del coche
year_actual = datetime.now().year
df["antiquity"] = year_actual - df["year"]

# Elimino los NaN.
df = df.dropna()

# Elimino todos coches con un precio mayor de 50 mil para quitarme Outliers.
coches_lujo = df['price'] > 50000
df_lujo = df[coches_lujo]
df = df.drop(df[coches_lujo].index)

# Unifico en solo dos posibilidades la columna de puertas.
df["doors"] = df["doors"].replace(5, 4)
df["doors"] = df["doors"].replace(3, 2)
df["doors"].unique()

# Guardo el DataFrame en un csv.
df.to_csv("../data/processed/coches_processed.csv", index=False)

# Separo el DataFrame en Train y Test, y los guardo en archivos CSV separados.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("../data/train/coches_train.csv", index=False)
test_df.to_csv("../data/test/coches_test.csv", index=False)