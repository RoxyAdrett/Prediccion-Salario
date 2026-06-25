import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

# Lista de columnas (DEBE ser idéntica a la que usaste en el entrenamiento)
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

# --- Mapeos para traducir (Español -> Formato del Modelo) ---
mapa_educacion = {'Licenciatura': 'Bachelor', 'Técnico': 'Diploma', 'Secundaria': 'High School', 'Máster': 'Master', 'Doctorado': 'PhD'}
mapa_industria = {'Consultoría': 'Consulting', 'Educación': 'Education', 'Finanzas': 'Finance', 'Gobierno': 'Government', 'Salud': 'Healthcare', 'Manufactura': 'Manufacturing', 'Medios': 'Media', 'Retail': 'Retail', 'Tecnología': 'Technology', 'Telecomunicaciones': 'Telecom'}
mapa_empresa = {'Empresa Grande': 'Enterprise', 'Grande': 'Large', 'Mediana': 'Medium', 'Pequeña': 'Small', 'Startup': 'Startup'}
mapa_ubicacion = {'Australia': 'Australia', 'Canadá': 'Canada', 'Alemania': 'Germany', 'India': 'India', 'Países Bajos': 'Netherlands', 'Remoto': 'Remote', 'Singapur': 'Singapore', 'Suecia': 'Sweden', 'Reino Unido': 'UK', 'EE.UU.': 'USA'}
mapa_remoto = {'Híbrido': 'Hybrid', 'No': 'No', 'Sí': 'Yes'}

# --- IMPORTANTE: Define aquí los puestos de trabajo y su valor codificado ---
mapa_puestos = {
    'Desarrollador Software': 120000.0, # Sustituye estos valores por los reales de tu modelo
    'Analista de Datos': 110000.0,
    'Gerente de Proyecto': 140000.0
}

# --- Interfaz ---
st.title("Demo: Predicción Salarial")

col1, col2 = st.columns(2)
with col1:
    exp = st.number_input("Años de experiencia", 0, 40, 5)
    educacion_sel = st.selectbox("Educación", list(mapa_educacion.keys()))
    industria_sel = st.selectbox("Industria", list(mapa_industria.keys()))
    remoto_sel = st.selectbox("Trabajo remoto", list(mapa_remoto.keys()))

with col2:
    skills = st.number_input("Cantidad de habilidades", 1, 20, 5)
    certs = st.number_input("Certificaciones", 0, 10, 0)
    empresa_sel = st.selectbox("Tamaño de empresa", list(mapa_empresa.keys()))
    ubicacion_sel = st.selectbox("Ubicación", list(mapa_ubicacion.keys()))

# Lista desplegable para el puesto
titulo_puesto_sel = st.selectbox("Título del puesto", list(mapa_puestos.keys()))

if st.button("Calcular Salario"):
    # 1. Crear DataFrame base con ceros
    input_df = pd.DataFrame(0, index=[0], columns=cols)
    
    # 2. Asignar valores numéricos
    input_df['experience_years'] = float(exp)
    input_df['skills_count'] = float(skills)
    input_df['certifications'] = float(certs)
    
    # 3. Asignar categorías seleccionadas
    input_df[f'education_level_{mapa_educacion[educacion_sel]}'] = 1.0
    input_df[f'industry_{mapa_industria[industria_sel]}'] = 1.0
    input_df[f'company_size_{mapa_empresa[empresa_sel]}'] = 1.0
    input_df[f'location_{mapa_ubicacion[ubicacion_sel]}'] = 1.0
    input_df[f'remote_work_{mapa_remoto[remoto_sel]}'] = 1.0
    
    # 4. Asignar valor del puesto
    input_df['job_title_encoded'] = mapa_puestos[titulo_puesto_sel]

    prediccion = model.predict(input_df)[0]
    st.success(f"### El salario estimado es: ${prediccion:,.2f} USD")
