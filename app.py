import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

st.title("Demo: Predicción Salarial")

exp = st.number_input("Años de experiencia", 0, 40, 5, 1)
skills = st.number_input("Cantidad de habilidades", 1, 20, 5, 1)

# Listas de opciones basadas en tu entrenamiento
educacion_options = ['Bachelor', 'Master', 'PhD', 'Diploma', 'High School']
industria_options = ['Technology', 'Finance', 'Healthcare', 'Consulting', 'Education']

educacion = st.selectbox("Nivel de educación", educacion_options)
industria = st.selectbox("Industria", industria_options)

if st.button("Calcular Salario"):
    # 1. Crear todas las columnas necesarias en 0
    cols = ['experience_years', 'skills_count', 'certifications', 'is_outlier_multivariate', 
            'education_level_Bachelor', 'education_level_Diploma', 'education_level_High School', 
            'education_level_Master', 'education_level_PhD', 'industry_Consulting', 'industry_Education', 
            'industry_Finance', 'industry_Government', 'industry_Healthcare', 'industry_Manufacturing', 
            'industry_Media', 'industry_Retail', 'industry_Technology', 'industry_Telecom', 
            'company_size_Enterprise', 'company_size_Large', 'company_size_Medium', 'company_size_Small', 
            'company_size_Startup', 'location_Australia', 'location_Canada', 'location_Germany', 
            'location_India', 'location_Netherlands', 'location_Remote', 'location_Singapore', 
            'location_Sweden', 'location_UK', 'location_USA', 'remote_work_Hybrid', 
            'remote_work_No', 'remote_work_Yes', 'job_title_encoded']
    
    input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    
    # 2. Asignar los valores del usuario
    input_df['experience_years'] = float(exp)
    input_df['skills_count'] = float(skills)
    input_df[f'education_level_{educacion}'] = 1.0
    input_df[f'industry_{industria}'] = 1.0
    
    # 3. Predicción
    resultado = model.predict(input_df)
    st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
