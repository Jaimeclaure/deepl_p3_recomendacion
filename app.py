import streamlit as st
import pandas as pd
import joblib
import random
import re
import requests
import urllib.parse

# 1. Configuración de la página
st.set_page_config(page_title='Proyecto de Deep Learning / Jaime Claure', page_icon='🎵', layout='wide')

# --- CONFIGURACIÓN DE ITUNES API (Alternativa libre a Spotify) ---
def get_album_cover(song_title, artist_name):
    # Imagen genérica confiable en caso de error
    fallback_url = "https://dummyimage.com/150x150/282828/1db954.png&text=No+Cover"
    
    try:
        # Limpieza de datos (Borramos paréntesis y corchetes)
        clean_title = re.sub(r'\(.*?\)|\[.*?\]', '', song_title).strip()
        clean_artist = re.sub(r'\(.*?\)|\[.*?\]', '', artist_name).strip()
        
        # Codificamos el texto para la URL
        query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        
        # Llamada a la API pública de iTunes
        url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Si iTunes encuentra la canción, extraemos la URL de la portada
        if data['resultCount'] > 0:
            # Obtenemos la imagen (por defecto 100x100px)
            return data['results'][0]['artworkUrl100']
        else:
            return fallback_url
    except Exception as e:
        print(f"Error en iTunes API: {e}")
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
st.title('Sistema de Recommendaciones | Proyecto de Deep Learning')
st.caption('Desarrollado por: **Jaime Claure**')

st.markdown("""
Este proyecto de Deep Learning realiza un analiza bajo tres enfoques:
* **Filtrado Colaborativo:** Agrupa a usuarios y canciones para encontrar patrones y predecir afinidades.
* **Procesamiento de Lenguaje Natural:** Analiza el contexto semantico basadas en contenido.
* **Redes Neuronales en Grafos:** Mapea conexiones de escucha para extraer embeddings de cada cancion.
""")

with st.expander("Como funciona la arquitectura detras del proyecto?"):
    st.markdown("""      
    Proyecto de Deep Learning diseñado para abordar los desafíos de los sistemas de recomendacion:
    
    1. Filtrado colaborativo que agrupa simultaneamente usuarios y canciones en clusters usando patrones de interaccion y predecir sus afinidades\n\n
    2. Utilizando procesamiento del lenguaje natural para medir la similitud del coseno y comparar el contexto semántico de las canciones, así recomendar a usuarios nuevos que no tienen historial suficiente de interacciones\n\n
    3. Redes neuronales con grafos de co-ocurrencia, donde Node2Vec explora el grafo de forma guiada para capturar similitudes estructurales y en PyTorch se entrenan embeddings más profundos del catálogo musical
    """)

try:
    df, model = load_data()    
    valid_users = sorted(df['user_id'].unique())    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selecciona un usuario")
        user_id_input = st.selectbox('Usuarios válidos en la base de datos:', valid_users)
        
        if st.button('Elige usuario al azar'):
            user_id_input = random.choice(valid_users)
            st.rerun()

    # Actualizado el argumento deprecado a width='stretch'
    if st.button('Analizar perfil y recomendar', width='stretch'):
        hist_col, rec_col = st.columns(2)
        with hist_col:
            st.subheader("Historial de escucha")
            st.caption("Las canciones mas reproducidas por el usuario")
            history_df = get_user_history(df, user_id_input)
            history_df = history_df.rename(columns={'title': 'Canción', 'artist_name': 'Artista', 'play_count': 'Reproducciones'})
            # Actualizado el argumento deprecado a width='stretch'
            st.dataframe(history_df, width='stretch', hide_index=True)
            
        with rec_col:
            st.subheader("Recomendaciones para vos")
            st.caption("Recomendaciones basadas en gustos similares:")
            with st.spinner('Haciendo la magia..'):
                rec_df = get_recommendations(df, user_id_input, model)
                
                # Desplegar tarjetas visuales con carátulas de Apple Music
                for index, row in rec_df.iterrows():
                    cancion = row['Canción']
                    artista = row['Artista']
                    score = row['Match Score']
                    
                    # Llamada a iTunes API
                    portada_url = get_album_cover(cancion, artista)
                    
                    # Maquetación de la tarjeta
                    img_col, txt_col = st.columns([1, 4])
                    with img_col:
                        st.image(portada_url, width=80)
                    with txt_col:
                        st.markdown(f"**{cancion}**")
                        st.write(f"{artista} | {score}")

except Exception as e:
    st.error(f'Error al cargar la aplicación: {e}')