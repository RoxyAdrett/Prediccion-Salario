import streamlit as st
import joblib
import pandas as pd

# Cargar el pipeline completo
@st.cache_resource
def load_model():
    # Asegúrate de que este archivo sea el pipeline completo, no solo el estimador
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

st.title("Demo: Predicción Salarial")

# Entradas del usuario
exp = st.number_input("Años de experiencia", 0.0, 40.0, 5.0)
skills = st.number_input("Cantidad de habilidades", 1.0, 20.0, 5.0)
educacion = st.selectbox("Educación", ['Bachelor', 'Master', 'PhD', 'Diploma', 'High School'])
industria = st.selectbox("Industria", ['Technology', 'Finance', 'Healthcare', 'Consulting', 'Education'])

if st.button("Calcular Salario"):
    # Crear un DataFrame con los valores originales (el pipeline se encarga de las 38 columnas)
    input_data = pd.DataFrame({
        'experience_years': [exp],
        'skills_count': [skills],
        'education_level': [educacion],
        'industry': [industria]
        # Agrega aquí las otras columnas que tu pipeline espera
    })
    
    # El pipeline transformará estos datos automáticamente a las 38 columnas
    resultado = model.predict(input_data)
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
