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
st.title('🎵 AI Music Recommender | Proyecto de Deep Learning')
st.caption('Desarrollado por: **Jaime Claure**')

st.markdown("""
Explora recomendaciones personalizadas impulsadas por un motor de Inteligencia Artificial. Este proyecto de investigación analiza el ecosistema musical utilizando tres enfoques avanzados:
* **Filtrado Colaborativo (Co-Clustering):** Agrupa simultáneamente a usuarios y canciones para encontrar patrones ocultos y predecir afinidades.
* **Procesamiento de Lenguaje Natural (TF-IDF):** Analiza el contexto semántico (título, artista, álbum) para sugerencias basadas en contenido.
* **Redes Neuronales en Grafos (GNN):** Mapea conexiones de escucha, utilizando *Random Walks* y arquitecturas *Skip-gram* en PyTorch para extraer *embeddings* profundos de cada pista.

*(Actualmente ejecutando en producción: Motor de Co-Clustering Optimizado).*
""")
st.divider()

with st.expander("🧠 Conoce la Ingeniería del Proyecto (Arquitectura de Datos)"):
    st.markdown("""
    **¿Cómo funciona el motor híbrido bajo el capó?**
    
    Este proyecto fue diseñado para abordar los desafíos clásicos de los sistemas de recomendación en producción, implementando tres estrategias de Deep Learning:
    
    1. **Filtrado Colaborativo (Co-Clustering):** *El motor principal.* Analiza patrones de comportamiento agrupando simultáneamente a usuarios y canciones. Es excelente para predecir ratings cuando hay abundante historial de interacciones.
    2. **Redes Neuronales en Grafos (GNN - Node2Vec):** *Descubrimiento profundo.* Mapea el ecosistema musical como un grafo. Utilizando Random Walks y arquitecturas Skip-gram, genera 'embeddings' que capturan relaciones complejas y no lineales entre pistas.
    3. **Content-Based (NLP + TF-IDF):** *La solución al 'Cold Start'.* Utiliza Procesamiento de Lenguaje Natural y Similitud del Coseno para analizar el contexto semántico de los metadatos. Es vital en producción para recomendar pistas recién lanzadas o atender a usuarios nuevos sin historial previo.
    """)

try:
    df, model = load_data()
    
    valid_users = sorted(df['user_id'].unique())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("👤 Selecciona un Usuario")
        user_id_input = st.selectbox('Usuarios válidos en la base de datos:', valid_users)
        
        if st.button('🎲 Elegir usuario al azar'):
            user_id_input = random.choice(valid_users)
            st.rerun()

    # Actualizado el argumento deprecado a width='stretch'
    if st.button('🚀 Analizar Perfil y Recomendar', width='stretch'):
        st.divider()
        
        hist_col, rec_col = st.columns(2)
        
        with hist_col:
            st.subheader("📻 Su Historial de Escuchas")
            st.caption("Las canciones que este usuario más ha reproducido:")
            history_df = get_user_history(df, user_id_input)
            history_df = history_df.rename(columns={'title': 'Canción', 'artist_name': 'Artista', 'play_count': 'Reproducciones'})
            # Actualizado el argumento deprecado a width='stretch'
            st.dataframe(history_df, width='stretch', hide_index=True)
            
        with rec_col:
            st.subheader("✨ Recomendaciones para Ti")
            st.caption("Nuestras sugerencias basadas en gustos similares:")
            with st.spinner('Conectando con la IA y obteniendo portadas...'):
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
                        st.write(f"🎤 {artista} | {score}")
                    
                    st.divider()

except Exception as e:
    st.error(f'Error al cargar la aplicación: {e}')