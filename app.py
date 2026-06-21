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

st.title("Demo: Predicción Salarial")

# --- SECCIÓN SUPERIOR: Predicción principal ---
col1, col2 = st.columns(2)
with col1:
    exp = st.number_input("Años de experiencia", 0, 40, 5, 1)
    educacion = st.selectbox("Educación", ['Bachelor', 'Master', 'PhD', 'Diploma', 'High School'])
with col2:
    skills = st.number_input("Cantidad de habilidades", 1, 20, 5, 1)
    industria = st.selectbox("Industria", ['Technology', 'Finance', 'Healthcare', 'Consulting', 'Education'])

if st.button("Calcular Salario Principal"):
    input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    input_df['experience_years'] = float(exp)
    input_df['skills_count'] = float(skills)
    input_df[f'education_level_{educacion}'] = 1.0
    input_df[f'industry_{industria}'] = 1.0
    
    st.session_state.res_principal = model.predict(input_df)[0]
    st.write(f"### El salario estimado es: ${st.session_state.res_principal:,.2f}")

st.divider()

# --- SECCIÓN INFERIOR: Test Comparativo ---
st.subheader("Test Comparativo de Escenarios")

def generar_test():
    # Simulamos 3 escenarios aleatorios
    scenarios = [
        {'exp': 2, 'skills': 2, 'edu': 'High School', 'ind': 'Education'},
        {'exp': 10, 'skills': 8, 'edu': 'Bachelor', 'ind': 'Technology'},
        {'exp': 20, 'skills': 15, 'edu': 'PhD', 'ind': 'Finance'}
    ]
    results = []
    for s in scenarios:
        df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        df['experience_years'] = float(s['exp'])
        df['skills_count'] = float(s['skills'])
        df[f"education_level_{s['edu']}"] = 1.0
        df[f"industry_{s['ind']}"] = 1.0
        val = model.predict(df)[0]
        results.append({"Escenario": f"{s['edu']} en {s['ind']} ({s['exp']} años)", "Salario": f"${val:,.2f}"})
    return results

if st.button("Test Nuevo"):
    st.session_state.test_data = generar_test()

if 'test_data' in st.session_state:
    st.table(pd.DataFrame(st.session_state.test_data))
