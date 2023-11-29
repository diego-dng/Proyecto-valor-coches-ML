import pandas as pd
import joblib
import streamlit as st

def predecir_precio(make, year, doors, kms, power):
    # antiquity = 2023 - year
    modelo = joblib.load('../models/modelo_KNeighborsRegressor.pkl')
    df_pred = pd.read_csv('../data/test/coches_test.csv')
    df_pred = df_pred.drop(df_pred.index)
    # x = df[["kms", "power", "antiquity", "doors"]]
    antiquity = 2023 - year
    nuevo_registro = {'antiquity': antiquity,'doors': doors, 'kms': kms, "power": power}

    # Añadir el nuevo registro al DataFrame
    df_pred = pd.concat([df_pred, pd.DataFrame([nuevo_registro])], ignore_index=True)
    df_pred = df_pred.dropna(axis=1)
    #df_pred
    # Preprocesamiento de datos (ajusta esto según el preprocesamiento real que hiciste durante el entrenamiento)
    #datos = pd.DataFrame({'Marca': [marca], 'Año': [año], 'Kilometraje': [kilometraje], 'Potencia': [potencia]})
        
    # Realizar la predicción utilizando el modelo cargado
    precio_predicho = modelo.predict(df_pred)
    precio_predicho = precio_predicho.round(2)
    precio_predicho = str(precio_predicho[0]) + "€"
    precio_predicho
    
    return precio_predicho

def formulario():
    st.title("Formulario y Predicción de Precio de Automóviles")

    # Recopilar información del usuario
    make = st.text_input("Marca del vehículo:")
    year = st.number_input("Año del vehículo:", max_value=2023, step=1)
    doors = st.number_input("Puertas del vehículo:")
    kms = st.number_input("Kilometraje del vehículo:")
    power = st.number_input("Potencia del vehículo:")

    # Botón para realizar la predicción
    if st.button("Predecir Precio"):
        # Realizar la predicción utilizando la función predefinida
        precio_predicho = predecir_precio(make, year, doors, kms, power)

        # Mostrar la información recopilada y la predicción
        st.success("Información Recopilada:")
        st.info(f"Marca: {make}")
        st.info(f"Año: {year}")
        st.info(f"Puertas: {doors}")
        st.info(f"Kilometraje: {kms} km")
        st.info(f"Potencia: {power} CV")
        st.success(f"Precio Predicho: {precio_predicho}")