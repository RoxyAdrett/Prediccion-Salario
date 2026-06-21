import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

# Lista de las 38 columnas necesarias
COLUMNS = ['experience_years', 'skills_count', 'certifications', 'is_outlier_multivariate', 
           'education_level_Bachelor', 'education_level_Diploma', 'education_level_High School', 
           'education_level_Master', 'education_level_PhD', 'industry_Consulting', 'industry_Education', 
           'industry_Finance', 'industry_Government', 'industry_Healthcare', 'industry_Manufacturing', 
           'industry_Media', 'industry_Retail', 'industry_Technology', 'industry_Telecom', 
           'company_size_Enterprise', 'company_size_Large', 'company_size_Medium', 'company_size_Small', 
           'company_size_Startup', 'location_Australia', 'location_Canada', 'location_Germany', 
           'location_India', 'location_Netherlands', 'location_Remote', 'location_Singapore', 
           'location_Sweden', 'location_UK', 'location_USA', 'remote_work_Hybrid', 
           'remote_work_No', 'remote_work_Yes', 'job_title_encoded']

st.title("Demo: Predicción Salarial")

# Entradas simples
exp = st.number_input("Años de experiencia", 0, 40, 5)
skills = st.number_input("Cantidad de habilidades", 1, 20, 5)
# Añade esto después de las variables 'exp' y 'skills'
educacion = st.selectbox("Nivel de educación", ['Bachelor', 'Master', 'PhD', 'Diploma'])
industria = st.selectbox("Industria", ['Technology', 'Finance', 'Healthcare', 'Consulting'])

# Para las otras variables, definimos valores por defecto (ej: 0)
# En una app real, deberías añadir más inputs para que el usuario elija, 
# pero esto hará que el modelo funcione ya mismo.
if st.button("Calcular Salario"):
    # Creamos un vector de ceros
    data = np.zeros((1, len(COLUMNS)))
    input_df = pd.DataFrame(data, columns=COLUMNS)
    
    # Asignamos las entradas del usuario
    input_df['experience_years'] = exp
    input_df['skills_count'] = skills
    
    # Aquí puedes añadir los valores por defecto que se usaron en el entrenamiento 
    # (por ejemplo, educación Bachelor=1)
    # En la lógica del botón, modifica la asignación:
    input_df[f'education_level_{educacion}'] = 1
    input_df[f'industry_{industria}'] = 1

    resultado = model.predict(input_df)
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
