
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from surprise import Reader, Dataset

# Configuración de la página
st.set_page_config(page_title='Recomendador de Música', layout='wide')

@st.cache_resource
def load_resources():
    df = pd.read_csv('songs_data_app.csv')
    model = joblib.load('model_colab.pkl')
    return df, model

def get_recommendations(df, user_id, model, n=5):
    # Obtener todas las canciones únicas
    all_songs = df['song_id'].unique()
    # Obtener canciones que el usuario ya escuchó
    user_songs = df[df['user_id'] == user_id]['song_id'].unique()
    # Canciones no escuchadas
    songs_to_predict = [s for s in all_songs if s not in user_songs]
    
    predictions = [model.predict(user_id, s_id) for s_id in songs_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n = predictions[:n]
    rec_ids = [p.iid for p in top_n]
    
    # Obtener títulos de las canciones
    return df[df['song_id'].isin(rec_ids)][['title', 'artist_name']].drop_duplicates()

# Interfaz de usuario
st.title('🎵 Sistema de Recomendación de Música')
df, model = load_resources()

user_id_input = st.number_input('Ingresa tu ID de Usuario (ejemplo: 6958)', min_value=0, step=1)

if st.button('Generar Recomendaciones'):
    if user_id_input in df['user_id'].unique():
        results = get_recommendations(df, user_id_input, model)
        st.subheader(f'Top 5 recomendaciones para el usuario {user_id_input}:')
        st.table(results)
    else:
        st.error('ID de usuario no encontrado en el dataset filtrado.')
