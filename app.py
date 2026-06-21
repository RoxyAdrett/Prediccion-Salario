import streamlit as st
import joblib
import pandas as pd

# Cargar tu modelo
model = joblib.load('mejor_modelo_salario.pkl')

st.title("Demo: Predicción Salarial")

# Entradas para el usuario
exp = st.number_input("Años de experiencia", min_value=0, max_value=40, value=5)
habilidades = st.number_input("Cantidad de habilidades", min_value=1, max_value=20, value=5)

if st.button("Calcular Salario"):
    # Asegúrate de que estos nombres ('experience_years', 'skills_count') 
    # sean EXACTAMENTE iguales a los que usaste al entrenar tu modelo
    input_data = pd.DataFrame({
        'experience_years': [exp], 
        'skills_count': [habilidades]
    })
    
    # Algunas versiones de LightGBM necesitan que el DataFrame 
    # tenga el mismo tipo de dato que el entrenamiento (ej: float)
    input_data = input_data.astype({'experience_years': 'float', 'skills_count': 'float'})

    resultado = model.predict(input_data)
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
