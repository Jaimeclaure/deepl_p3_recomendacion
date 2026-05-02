import streamlit as st
import pandas as pd
import joblib
import random

# 1. Configuración de la página más atractiva
st.set_page_config(page_title='Proyecto de Deep Learning / Jaime Claure/Sistema de Recommendacion', page_icon='', layout='wide')

@st.cache_resource
def load_data():
    df = pd.read_csv('songs_data_app.csv')
    model = joblib.load('model_colab.pkl')
    return df, model

# Función para obtener el historial del usuario
def get_user_history(df, user_id, n=5):
    # Filtramos por el usuario y ordenamos por las canciones que más veces reprodujo
    user_history = df[df['user_id'] == user_id].sort_values(by='play_count', ascending=False)
    # Seleccionamos las columnas relevantes y quitamos duplicados
    return user_history[['title', 'artist_name', 'play_count']].drop_duplicates(subset=['title']).head(n)

# Función mejorada de recomendaciones
def get_recommendations(df, user_id, model, n=5):
    all_songs = df['song_id'].unique()
    user_songs = df[df['user_id'] == user_id]['song_id'].unique()
    songs_to_predict = [s for s in all_songs if s not in user_songs]

    predictions = [model.predict(user_id, s_id) for s_id in songs_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Construimos una lista de diccionarios para hacer un DataFrame más bonito
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
st.title('Proyecto de Deep Learning / Jaime Claure / Sistema de Recomendación de Música')
st.markdown('Explora recomendaciones personalizadas impulsadas por un motor de Inteligencia Artificial (Co-Clustering).')
st.divider()

try:
    df, model = load_data()
    
    # Extraer lista de usuarios válidos
    valid_users = sorted(df['user_id'].unique())
    
    # 2. Selector de usuario interactivo
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selecciona un Usuario")
        user_id_input = st.selectbox('Usuarios válidos en la base de datos:', valid_users)
        
        # Botón para elegir un usuario al azar
        if st.button('Elegir usuario al azar'):
            user_id_input = random.choice(valid_users)
            st.rerun()

    # Botón principal de ejecución
    if st.button('Analizar Perfil y Recomendar', use_container_width=True):
        st.divider()
        
        # 3. Layout de dos columnas para comparar el Antes y el Después
        hist_col, rec_col = st.columns(2)
        
        with hist_col:
            st.subheader("Su Historial de Escuchas")
            st.caption("Las canciones que este usuario más ha reproducido:")
            history_df = get_user_history(df, user_id_input)
            
            # Renombrar columnas para la interfaz
            history_df = history_df.rename(columns={'title': 'Canción', 'artist_name': 'Artista', 'play_count': 'Reproducciones'})
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
        with rec_col:
            st.subheader("Recomendaciones para Ti")
            st.caption("Nuestras sugerencias basadas en gustos similares:")
            with st.spinner('Analizando millones de conexiones...'):
                rec_df = get_recommendations(df, user_id_input, model)
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f'Error al cargar la aplicación: {e}')