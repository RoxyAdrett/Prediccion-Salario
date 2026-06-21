import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

st.title("Demo: Predicción Salarial")

# Ajuste para números enteros (step=1)
exp = st.number_input("Años de experiencia", min_value=0, max_value=40, value=5, step=1)
skills = st.number_input("Cantidad de habilidades", min_value=1, max_value=20, value=5, step=1)

# Agregué selectores para que el modelo tenga datos distintos y cambie el resultado
educacion = st.selectbox("Nivel de educación", ['Bachelor', 'Master', 'PhD', 'Diploma', 'High School'])
industria = st.selectbox("Industria", ['Technology', 'Finance', 'Healthcare', 'Consulting', 'Education'])

if st.button("Calcular Salario"):
    # Creamos el DataFrame con los datos enteros
    input_data = pd.DataFrame({
        'experience_years': [int(exp)],
        'skills_count': [int(skills)],
        'education_level': [educacion],
        'industry': [industria]
    })
    
    # El pipeline hará la transformación a las 38 columnas automáticamente
    resultado = model.predict(input_data)
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
