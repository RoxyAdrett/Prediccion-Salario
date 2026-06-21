import streamlit as st
import joblib
import pandas as pd

# Cargar tu modelo
model = joblib.load('mejor_modelo_salario (1).pkl')

st.title("Demo: Predicción Salarial")

# Entradas para el usuario
exp = st.number_input("Años de experiencia", min_value=0, max_value=40, value=5)
habilidades = st.number_input("Cantidad de habilidades", min_value=1, max_value=20, value=5)

if st.button("Calcular Salario"):
    # Creamos un pequeño dataframe con los datos
    input_data = pd.DataFrame({'experience_years': [exp], 'skills_count': [habilidades]})
    
    # Aquí el modelo predice
    resultado = model.predict(input_data)
    
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")