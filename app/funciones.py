import pandas as pd
import joblib
import streamlit as st

def predecir_precio(make, year, doors, kms, power):
   
    modelo = joblib.load('../models/modelo_RandomForestRegressor/modelo_RandomForestRegressor.pkl')
    
    antiquity = 2023 - year
    nuevo_registro = {'antiquity': antiquity,'doors': doors, 'kms': kms, "power": power}
    df_pred = pd.DataFrame([nuevo_registro])

    print(df_pred)
    
    precio_predicho = modelo.predict(df_pred)
    print("el precio es:", precio_predicho)
    precio_predicho = precio_predicho.round(2)
    precio_predicho = str(precio_predicho[0]) + "€"
    precio_predicho
    
    return precio_predicho

def formulario():
    

    make = st.text_input("Marca del vehículo:")
    year = st.number_input("Año del vehículo:", max_value=2023, step=1)
    doors = st.number_input("Puertas del vehículo:")
    kms = st.number_input("Kilometraje del vehículo:")
    power = st.number_input("Potencia del vehículo:")


    if st.button("Predecir Precio"):
        precio_predicho = predecir_precio(make, year, doors, kms, power)

        st.success("Información Recopilada:")
        st.info(f"Marca: {make}")
        st.info(f"Año: {year}")
        st.info(f"Puertas: {doors}")
        st.info(f"Kilometraje: {kms} km")
        st.info(f"Potencia: {power} CV")
        st.success(f"Precio Predicho: {precio_predicho}")