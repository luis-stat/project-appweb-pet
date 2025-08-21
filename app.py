import streamlit as st
import numpy as np
import joblib

model = joblib.load("modelo_treinado.pkl")

st.title("🌸 Classificador de Flores Iris")

sepal_length = st.slider("Comprimento da Sépala (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Largura da Sépala (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Comprimento da Pétala (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Largura da Pétala (cm)", 0.1, 2.5, 1.0)

if st.button("🔍 Verificar Espécie"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    
    st.subheader("Espécie prevista:")
    st.success(prediction)
