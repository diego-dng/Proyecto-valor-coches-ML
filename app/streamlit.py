# Importo librerías
#from funct import funciones as fun
# from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import subprocess
import streamlit as st
import pandas as pd
import joblib

import funciones

# Título de la página
st.set_page_config(page_title = 'valorDeTuCoche.net', page_icon= ":car:")
st.image('img/concesionario.jpg')
st.title('Valor de un coche:')
st.divider()

# Datasets a trabajar
coches_raw = pd.read_csv('../data/raw/coches-de-segunda-mano-sample.csv')
coches_processed = pd.read_csv('../data/processed/coches_processed.csv')
coches_train = pd.read_csv('../data/train/coches_train.csv')
cohes_test = pd.read_csv('../data/test/coches_test.csv')

funciones.formulario()  
    



