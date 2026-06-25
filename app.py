import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

# Lista de columnas (las mismas 38)
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

# Diccionarios de mapeo para traducir la selección al formato que entiende el modelo
mapa_educacion = {
    'Licenciatura / Grado universitario': 'Bachelor',
    'Máster / Maestría': 'Master',
    'Doctorado': 'PhD',
    'Diplomatura / Título técnico': 'Diploma',
    'Enseñanza media / Secundaria completa': 'High School'
}

mapa_industria = {
    'Tecnología': 'Technology',
    'Finanzas': 'Finance',
    'Salud / Servicios sanitarios': 'Healthcare',
    'Consultoría': 'Consulting',
    'Educación': 'Education'
}

st.title("Demo: Predicción Salarial")

# --- SECCIÓN SUPERIOR ---
col1, col2 = st.columns(2)
with col1:
    exp = st.number_input("Años de experiencia", 0, 40, 5, 1)
    educacion_esp = st.selectbox("Educación", list(mapa_educacion.keys()))
with col2:
    skills = st.number_input("Cantidad de habilidades", 1, 20, 5, 1)
    industria_esp = st.selectbox("Industria", list(mapa_industria.keys()))

if st.button("Calcular Salario Principal"):
    # Convertimos la selección de usuario al valor que espera el modelo
    educacion_en = mapa_educacion[educacion_esp]
    industria_en = mapa_industria[industria_esp]
    
    input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    input_df['experience_years'] = float(exp)
    input_df['skills_count'] = float(skills)
    
    # Usamos los valores en inglés para activar la columna correcta
    input_df[f'education_level_{educacion_en}'] = 1.0
    input_df[f'industry_{industria_en}'] = 1.0
    
    st.session_state.res_principal = model.predict(input_df)[0]
    st.write(f"### El salario estimado es: ${st.session_state.res_principal:,.2f}")

st.divider()
