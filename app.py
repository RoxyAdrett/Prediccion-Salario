import streamlit as st
import joblib
import pandas as pd

# 1. Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load('mejor_modelo_salario.pkl')

model = load_model()

st.title("Demo: Predicción Salarial")

# 2. Entradas del usuario
exp = st.number_input("Años de experiencia", min_value=0.0, max_value=40.0, value=5.0)
habilidades = st.number_input("Cantidad de habilidades", min_value=1.0, max_value=20.0, value=5.0)

# 3. Predicción
if st.button("Calcular Salario"):
    # IMPORTANTE: El orden de las columnas debe ser idéntico al de tu entrenamiento.
    # Si al entrenar usaste: X = df[['experience_years', 'skills_count']]
    # Entonces el orden debe ser:
    input_data = pd.DataFrame({
        'experience_years': [float(exp)],
        'skills_count': [float(habilidades)]
    })
    
    try:
        resultado = model.predict(input_data)
        st.write(f"### El salario estimado es: ${resultado[0]:,.2f}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.write("Verifica que los nombres de las columnas coincidan con los del entrenamiento.")
