import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

# --- 1. Definición de columnas esperadas por el modelo ---
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

# --- 2. Interfaz de Usuario ---
st.title("Demo: Predicción Salarial")

# Inputs numéricos
exp = st.number_input("Años de experiencia", 0, 40, 5)
skills = st.number_input("Cantidad de habilidades", 1, 20, 5)
certs = st.number_input("Certificaciones", 0, 10, 0)

# Selectboxes para categóricas
educacion = st.selectbox("Educación", ['Bachelor', 'Diploma', 'High School', 'Master', 'PhD'])
industria = st.selectbox("Industria", ['Consulting', 'Education', 'Finance', 'Government', 'Healthcare', 'Manufacturing', 'Media', 'Retail', 'Technology', 'Telecom'])
empresa = st.selectbox("Tamaño de empresa", ['Enterprise', 'Large', 'Medium', 'Small', 'Startup'])
ubicacion = st.selectbox("Ubicación", ['Australia', 'Canada', 'Germany', 'India', 'Netherlands', 'Remote', 'Singapore', 'Sweden', 'UK', 'USA'])
remoto = st.selectbox("Trabajo remoto", ['Hybrid', 'No', 'Yes'])

# Para job_title, necesitas el diccionario de codificación que usaste en el entrenamiento
# Ejemplo: job_map = {'Data Scientist': 120000, 'Developer': 90000}
titulo_puesto = st.text_input("Título del puesto (ej: Data Scientist)")

if st.button("Calcular Salario"):
    # --- 3. Preparación del DataFrame ---
    # Creamos un DF con ceros para todas las columnas
    input_df = pd.DataFrame(0, index=[0], columns=cols)
    
    # Asignamos valores numéricos
    input_df['experience_years'] = float(exp)
    input_df['skills_count'] = float(skills)
    input_df['certifications'] = float(certs)
    
    # Activamos la categoría seleccionada (One-Hot Encoding)
    input_df[f'education_level_{educacion}'] = 1.0
    input_df[f'industry_{industria}'] = 1.0
    input_df[f'company_size_{empresa}'] = 1.0
    input_df[f'location_{ubicacion}'] = 1.0
    input_df[f'remote_work_{remoto}'] = 1.0
    
    # --- 4. Manejo del Target Encoding para Job Title ---
    # IMPORTANTE: Aquí debes poner el valor numérico calculado en tu entrenamiento
    # Si no sabes el valor, el modelo fallará o dará un resultado incorrecto
    input_df['job_title_encoded'] = 100000.0 # Reemplaza por tu mapeo real

    prediccion = model.predict(input_df)[0]
    st.success(f"### Salario estimado: ${prediccion:,.2f} USD")
