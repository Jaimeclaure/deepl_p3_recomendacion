import streamlit as st
import pandas as pd
import joblib
import random
import re
import requests
import urllib.parse

# 1. Config
st.set_page_config(page_title='Proyecto de Deep Learning / Jaime Claure', page_icon='🎵', layout='wide')

# iTunes API
def get_album_cover(song_title, artist_name):
    # Imagen genérica confiable en caso de error
    fallback_url = "https://dummyimage.com/150x150/282828/1db954.png&text=No+Cover"
    
    try:
        # Limpieza de datos 
        clean_title = re.sub(r'\(.*?\)|\[.*?\]', '', song_title).strip()
        clean_artist = re.sub(r'\(.*?\)|\[.*?\]', '', artist_name).strip()
        
        # Codificamos el texto para la URL
        query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        
        # Llamada a la API pública de iTunes
        url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # iTunes encuentra la canción, extra el cover
        if data['resultCount'] > 0:
            # cover al 100x100
            return data['results'][0]['artworkUrl100']
        else:
            return fallback_url
    except Exception as e:
        print(f"Error en iTunes API: {e}")
        return fallback_url

# el modelo
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

# GUIUX
st.title('Sistema de Recomendaciones | Deep Learning', anchor=False)
st.caption('Proyecto desarrollado por: **Jaime Claure**')

st.markdown("""
Este proyecto de Deep Learning aborda los retos actuales en sistemas de recomendación mediante tres enfoques técnicos: un Filtrado Colaborativo que agrupa usuarios y canciones mediante Co-Clustering para predecir afinidades, un modelo de Procesamiento de Lenguaje Natural que emplea similitud del coseno para analizar el contexto semántico de los metadatos y el uso de Redes Neuronales en Grafos con Node2Vec y PyTorch para extraer vectores profundos basados en la estructura del catálogo musical.
""")

with st.expander("Observaciones y conclusiones.."):
    st.markdown("""      
### 1. Análisis y comparativa de modelos
Evalué las arquitecturas para definir su impacto en la experiencia de escucha, donde el Filtrado Colaborativo demostró ser el más preciso para usuarios frecuentes con un RMSE de $1.0487$, aunque presenta limitaciones ante la falta de historial, por su parte, el modelo basado en contenido resolvió mejor el "arranque en frío" mediante metadatos, mientras que las GNN destacaron por capturar conexiones de co-ocurrencia complejas, a pesar de requerir un mayor coste de procesamiento se recurrio a Colab.

\n \n

### 2. Propuesta de arquitectura final
Decidí implementar un sistema híbrido que prioriza el Co-Clustering como motor de personalización y utiliza el análisis de contenido como mecanismo de respaldo para nuevos usuarios asi poder garantizar un descubrimiento musical verdadero, tambien integré una corrección de popularidad que mitiga el sesgo hacia los éxitos comerciales, logrando un equilibrio entre precisión y variedad en las recomendaciones.

\n \n

### 3. Margen de mejora y trabajo para futuros proyectos
Para escalar el DeepLearning, podria integrar factorización matricial para refinar las predicciones y sustituir el análisis semántico actual por embeddings de lenguaje profundo como BERT, otro aspecto a entrenar/ajustar el modelo, también se podria mejorar el modelo cambiando el recorrido del grafo guiado de p(afecta la tendencia de revisitar el nodo anterior) y q (afecta la tendencia de explroar alrededor), permitiendo que la red identifique relaciones estructurales de 2do orden aun mas sutiles en el catalogo musical.
    """)

try:
    df, model = load_data()    
    valid_users = sorted(df['user_id'].unique())    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selecciona un usuario", anchor=False)
        user_id_input = st.selectbox('Usuarios válidos en la base de datos:', valid_users)
        
        if st.button('Elige usuario al azar'):
            user_id_input = random.choice(valid_users)
            st.rerun()

    
    if st.button('Analizar perfil y recomendar', width='stretch'):
        hist_col, rec_col = st.columns(2)
        with hist_col:
            st.subheader("Historial de escucha", anchor=False)
            st.caption("Las canciones mas reproducidas por el usuario")
            history_df = get_user_history(df, user_id_input)
            history_df = history_df.rename(columns={'title': 'Canción', 'artist_name': 'Artista', 'play_count': 'Reproducciones'})
            
            st.dataframe(history_df, width='stretch', hide_index=True)
            
        with rec_col:
            st.subheader("Recomendaciones para vos", anchor=False)
            st.caption("Recomendaciones basadas en gustos similares:")
            with st.spinner('Haciendo la magia..'):
                rec_df = get_recommendations(df, user_id_input, model)
                
                # covers de Apple Music
                for index, row in rec_df.iterrows():
                    cancion = row['Canción']
                    artista = row['Artista']
                    score = row['Match Score']
                    
                    # iTunes API
                    portada_url = get_album_cover(cancion, artista)
                    
                    # Maquetacion de la tarjeta
                    img_col, txt_col = st.columns([1, 4])
                    with img_col:
                        st.image(portada_url, width=80)
                    with txt_col:
                        st.markdown(f"**{cancion}**")
                        st.write(f"{artista} | {score}")

except Exception as e:
    st.error(f'Error al cargar la aplicación: {e}')