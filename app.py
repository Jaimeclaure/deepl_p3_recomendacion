import streamlit as st
import pandas as pd
import joblib
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 1. Configuración de la página
st.set_page_config(page_title='Proyecto de Deep Learning / Jaime Claure', page_icon='🎵', layout='wide')

# --- CONFIGURACIÓN SEGURA DE SPOTIFY API ---
# Streamlit leerá las credenciales de manera invisible desde su bóveda de "Secrets"
try:
    CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
    CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
except Exception as e:
    st.warning("Aviso: Las credenciales de Spotify no están configuradas en los Secrets de Streamlit. Las portadas no cargarán.")
    sp = None

# Función para buscar la portada del álbum en Spotify
def get_album_cover(song_title, artist_name):
# Imagen genérica tipo "carátula" que nunca será bloqueada por los navegadores
    fallback_url = "https://dummyimage.com/150x150/282828/1db954.png&text=No+Cover"
    
    if sp is None:
        return fallback_url
    
    try:
        # Búsqueda FLEXIBLE: sin etiquetas estrictas. Es mucho más precisa con el dataset.
        query = f"{song_title} {artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        # Verificamos que haya resultados y que el álbum tenga al menos una imagen
        if results['tracks']['items'] and len(results['tracks']['items'][0]['album']['images']) > 0:
            # Tomamos la imagen principal [0] para asegurar que exista
            image_url = results['tracks']['items'][0]['album']['images'][0]['url']
            return image_url
        else:
            return fallback_url
    except Exception as e:
        print(f"Error en Spotify: {e}")
        return fallback_url

# --- FUNCIONES DE DATOS Y MODELO ---
@st.cache_resource
def load_data():
    df = pd.read_csv('songs_data_app.csv')
    model = joblib.load('model_colab.pkl')
    return df, model

def get_user_history(df, user_id, n=5):
    user_history = df[df['user_id'] == user_id].sort_values(by='play_count', ascending=False)
    return user_history[['title', 'artist_name', 'play_count']].drop_duplicates(subset=['title']).head(n)

def get_recommendations(df, user_id, model, n=5):
    all_songs = df['song_id'].unique()
    user_songs = df[df['user_id'] == user_id]['song_id'].unique()
    songs_to_predict = [s for s in all_songs if s not in user_songs]

    predictions = [model.predict(user_id, s_id) for s_id in songs_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    rec_list = []
    for p in predictions[:n]:
        song_info = df[df['song_id'] == p.iid].iloc[0]
        rec_list.append({
            'Canción': song_info['title'],
            'Artista': song_info['artist_name'],
            'Match Score': f"{round(p.est, 2)} ⭐"
        })
    
    return pd.DataFrame(rec_list)

# --- INTERFAZ DE USUARIO ---
st.title('Sistema de Recomendación de Música - Proyecto de Deep Learning')
st.markdown(
    "Para el proyecto desarrollado por Jaime Claure, se emplea tres enfoques:\n\n"
    "1.- Filtrado colaborativo que agrupa simultaneamente usuarios y canciones en clusters usando patrones de interaccion y predecir sus afinidades\n\n"
    "2.- Utilizando procesamiento del lenguaje natural para medir la similitud del coseno y comparar el contexto semántico de las canciones, así recomendar a usuarios nuevos que no tienen historial suficiente de interacciones\n\n"
    "3.- Redes neuronales con grafos de co-ocurrencia, donde Node2Vec explora el grafo de forma guiada para capturar similitudes estructurales y en PyTorch se entrenan embeddings más profundos del catálogo musical"
)
st.divider()

try:
    df, model = load_data()
    
    valid_users = sorted(df['user_id'].unique())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selecciona un Usuario")
        user_id_input = st.selectbox('Usuarios válidos en la base de datos:', valid_users)
        
        if st.button('Elegir usuario al azar'):
            user_id_input = random.choice(valid_users)
            st.rerun()

    if st.button('Analizar Perfil y Recomendar', use_container_width=True):
        st.divider()
        
        hist_col, rec_col = st.columns(2)
        
        with hist_col:
            st.subheader("Su historial de musica")
            st.caption("Canciones mas reproducidas por el usuario:")
            history_df = get_user_history(df, user_id_input)
            history_df = history_df.rename(columns={'title': 'Canción', 'artist_name': 'Artista', 'play_count': 'Reproducciones'})
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
        with rec_col:
            st.subheader("Recomendaciones para vos")
            st.caption("Recomendacion basada en gustos similares:")
            with st.spinner('Conectando con la IA y Spotify API...'):
                rec_df = get_recommendations(df, user_id_input, model)
                
                # Desplegar tarjetas visuales con carátulas de Spotify
                for index, row in rec_df.iterrows():
                    cancion = row['Canción']
                    artista = row['Artista']
                    score = row['Match Score']
                    
                    # Llamada a Spotify
                    portada_url = get_album_cover(cancion, artista)
                    
                    # Maquetación de la tarjeta
                    img_col, txt_col = st.columns([1, 4])
                    with img_col:
                        st.image(portada_url, width=80)
                    with txt_col:
                        st.markdown(f"**{cancion}**")
                        st.write(f"🎤 {artista} | {score}")
                    
                    st.divider() # Separador visual

except Exception as e:
    st.error(f'Error al cargar la aplicación: {e}')