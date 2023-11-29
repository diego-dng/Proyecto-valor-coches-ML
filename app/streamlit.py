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
st.set_page_config(page_title = 'Valor coche', page_icon= ":car:")
st.title('Valor de un coche:')
st.header('Una predicción realizada con Machine Learning')
#st.image('https://sepuedeonosepuede.com/wp-content/uploads/2022/10/comer-granos-de-cafe-scaled.jpg')
st.divider()

# Datasets a trabajar
coches_raw = pd.read_csv('../data/raw/coches-de-segunda-mano-sample.csv')
coches_processed = pd.read_csv('../data/processed/coches_processed.csv')
coches_train = pd.read_csv('../data/train/coches_train.csv')
cohes_test = pd.read_csv('../data/test/coches_test.csv')

st.sidebar.title('valorDeTuCoche.net')
st.sidebar.divider()
# ¿Agua Sucia o Café
if st.sidebar.button('Inicio.'):
    st.title('Descripción del modelo.')
    st.markdown('Con este modelo podremos predecir el valor de un coche en función de sus caracteristicas.')
    
if st.sidebar.button('Datos.'):
    st.title('Aqui meto los datos')
    st.markdown('en esta zona se muestran los datos')

if st.sidebar.button('Modelo.'):
    st.title('Ejemplo de modelo')
    st.markdown('Meter una tabla con las prediciones del modelo.')

if st.sidebar.button('Pon a prueba nuestro modelo.'):
    st.title('Introduce los datos del coche:')
    
funciones.formulario()  


