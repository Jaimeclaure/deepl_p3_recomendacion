app_code_cloud = """
import streamlit as st
import pandas as pd
import joblib
import os
from surprise import Reader, Dataset

# Configuración de la página
st.set_page_config(page_title='Music Recommender', page_icon='🎵')

@st.cache_resource
def load_data():
    # Usamos rutas relativas para que funcione en cualquier entorno (Local o Cloud)
    df = pd.read_csv('songs_data_app.csv')
    model = joblib.load('model_colab.pkl')
    return df, model

def get_recommendations(df, user_id, model, n=5):
    all_songs = df['song_id'].unique()
    user_songs = df[df['user_id'] == user_id]['song_id'].unique()
    songs_to_predict = [s for s in all_songs if s not in user_songs]

    predictions = [model.predict(user_id, s_id) for s_id in songs_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    rec_ids = [p.iid for p in predictions[:n]]
    return df[df['song_id'].isin(rec_ids)][['title', 'artist_name']].drop_duplicates()

st.title('🎵 Sistema de Recomendación de Música')
st.markdown('Genera recomendaciones personalizadas basadas en el modelo de Co-Clustering.')

try:
    df, model = load_data()
    user_id_input = st.number_input('Introduce tu ID de Usuario:', min_value=int(df['user_id'].min()), max_value=int(df['user_id'].max()), step=1)

    if st.button('Obtener Recomendaciones'):
        if user_id_input in df['user_id'].unique():
            with st.spinner('Calculando mejores canciones para ti...'):
                res = get_recommendations(df, user_id_input, model)
                st.table(res)
        else:
            st.error('El usuario no existe en el dataset filtrado.')
except Exception as e:
    st.error(f'Error al cargar los recursos: {e}')
    st.info('Asegúrate de que model_colab.pkl y songs_data_app.csv estén en la misma carpeta que este script.')
"""

with open('app.py', 'w') as f:
    f.write(app_code_cloud)

print("Archivo app.py actualizado para despliegue en GitHub/Cloud.")