# Importar las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from streamlit_option_menu import option_menu
import streamlit.components.v1 as c

st.set_page_config(page_title="Coches segunda mano",
                   page_icon=":electric_plug:")

seleccion = st.sidebar.selectbox("Selecciona menu", ['Inicio','Modelos'])

if seleccion == "Inicio":
    st.title("Predicción del valor de coches de segunda mano.")
    #img = Image.open("")
    #st.image(img)
   

    with st.expander("Tabla"):
        df = pd.read_csv("../data/raw/coches-de-segunda-mano-sample.csv", sep=",")
        st.write(df.head())


