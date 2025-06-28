'''
CMD:
#pip install streamlit
#streamlit hello - para comprobar si esta instalado
-te pedira el email, pero puedes apretar enter para no ingresar correo 

Para correr el proyecto y poder visualizarlo en el formato se debe ejecutar 
el siguiente comando en la cmd del proyecto:
#streamlit run chatbot.py
'''
import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
import joblib
from collections import defaultdict

# Inicializaciones
nlp = spacy.load("es_core_news_sm")
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# Cargar modelos y datos
# ---------------------------
scaler = joblib.load("models/scaler.pkl")
reducer = joblib.load("models/umap_reducer.pkl")
clf = joblib.load("models/rf_classifier.pkl")

features = [
    'iglesias', 'resorts', 'playas', 'parques', 'teatros', 'museos',
    'centros comerciales', 'zoológicos', 'restaurantes', 'pubs/bares',
    'hamburgueserías/pizzerías', 'galerías de arte', 'discotecas',
    'piscinas', 'gimnasios', 'pastelerías', 'belleza y spas', 'cafeterías',
    'miradores', 'monumentos', 'jardines'
]


lugares_por_categoria = {
    "Gastronomía": [
        "Burger Home", "Shawarma Pizza & Grill", "Dublin Irish Pub", "Bar La Vida",
        "Entre Barros", "Central Inka", "Candela Bistró", "Suau", "Tierra de Fuego",
        "Divino Pecado", "Roof Burger", "Cafeterías", "Pastelerías"
    ],
    "Naturaleza y Parques": [
        "Jardín Botánico", "Parque Quinta Vergara", "Laguna Sausalito", "Jardín Botánico Nacional"
    ],
    "Playas y Miradores": [
        "Playa Las Salinas", "Playa El Sol", "Sector Cochoa", "Mirador Cochoa", "Mirador Pablo Neruda"
    ],
    "Cultura": [
        "Museo Artequin", "Museo Fonck", "Museo de Artes decorativas Palacio Rioja", 
        "Teatro municipal de Viña del Mar", "Sala Viña del Mar (ex Cine Arte)",
        "Casa de piedra", "Galería Tarquinia"
    ],
    "Compras": [
        "Mall Marina", "Mall Marina Arauco", "Espacio Urbano", "Espacio Urbano Viña Centro"
    ],
    "Aventura / Fauna": [
        "Zoológico de Quilpué", "Mundo reptil"
    ],
    "Alojamiento": [
        "Hotel Castillo", "Hotel del Mar", "Pacific Sunset House"
    ],
    "Monumentos / Plazas": [
        "Reloj de Flores", "Plaza Vergara", "Muelle Vergara"
    ]
}

recomendaciones_por_cluster = {
    0: ["Cultura", "Naturaleza y Parques", "Monumentos y Plazas"],
    1: ["Compras", "Gastronomía", "Cultura"],
    2: ["Alojamiento", "Playas y Miradores", "Cultura"],
    3: ["Gastronomía", "Cultura", "Compras"],
    4: ["Gastronomía", "Compras"],
    5: ["Gastronomía", "Fauna y Aventura", "Compras"],
    6: ["Playas y Miradores", "Cultura", "Naturaleza y Parques", "Compras"],
    7: ["Gastronomía", "Cultura"]
}


faq_categoria_to_interes = {
    "Actividades": [
        "parques", "zoológicos", "resorts", "teatros", "museos", 
        "centros comerciales", "galerías de arte", "piscinas", 
        "gimnasios", "belleza y spas"
    ],
    "Eventos": [
        "teatros", "galerías de arte", "discotecas"
    ],
    "Ubicación": [
        "miradores", "zoológicos", "iglesias", "playas", "parques", 
        "centros comerciales", "monumentos", "jardines"
    ],
    "Gastronomía": [
        "restaurantes", "hamburgueserías/pizzerías", "pubs/bares", 
        "cafeterías", "pastelerías"
    ],
    "Alojamientos": [
        "resorts"
    ]
}

sinonimos = {
    "playa": "playas",
    "museo": "museos",
    "iglesia": "iglesias",
    "parque": "parques",
    "teatro": "teatros",
    "galeria": "galerías de arte",
    "galería": "galerías de arte",
    "discoteca": "discotecas",
    "jardin": "jardines",
    "jardín": "jardines",
    "monumento": "monumentos",
    "zoologico": "zoológicos",
    "zoológico": "zoológicos",
    "restaurante": "restaurantes",
    "bar": "pubs/bares",
    "pub": "pubs/bares",
    "pizza": "hamburgueserías/pizzerías",
    "hamburguesa": "hamburgueserías/pizzerías",
    "piscina": "piscinas",
    "gimnasio": "gimnasios",
    "panaderia": "pastelerías",
    "cafeteria": "cafeterías",
    "cafe": "cafeterías",
    "belleza": "belleza y spas",
    "spa": "belleza y spas",
    "mirador": "miradores",
    "resort": "resorts"
}


stop_words = set(stopwords.words('spanish'))

def limpieza(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and len(token) > 1]
    return ' '.join(tokens)

def responder(pregunta_usuario):
    pregunta_proc = limpieza(pregunta_usuario)
    pregunta_vec = vectorizer.transform([pregunta_proc])
    similitudes = cosine_similarity(pregunta_vec, tfidf_matrix)
    max_similitud = similitudes.max()

    if max_similitud > np.percentile(similitudes, 75): 
        idx = similitudes.argmax()
        respuesta_texto = df_faq.iloc[idx]['respuesta']
        categoria = df_faq.iloc[idx]['categoría']

        # Aumentar intereses desde la categoría de la respuesta
        intereses = faq_categoria_to_interes.get(categoria, [])
        for interes in intereses:
            st.session_state.intereses_usuario[interes] += 1

        return respuesta_texto, categoria, round(float(max_similitud), 2)
    else:
        return "Lo siento, no tengo información sobre eso. ¿Podrías reformular tu pregunta?", None, round(float(max_similitud), 2)



def recomendar_por_intereses(intereses_dict):
    # Vector para el clasificador
    vector = [intereses_dict.get(f, 0) for f in features]
    vector_scaled = scaler.transform([vector])
    cluster = clf.predict(vector_scaled)[0]

    # Obtener categorías del cluster
    categorias = recomendaciones_por_cluster.get(cluster, [])

    resultado = []
    for categoria in categorias:
        lugares = lugares_por_categoria.get(categoria, [])
        if lugares:
            resultado.append(f"**{categoria}**:\n" + "\n".join(f"• {l}" for l in lugares))

    return resultado or ["No se encontraron recomendaciones específicas para tus gustos, pero aquí tienes algunas opciones generales."]


def extraer_intereses(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    actualizaciones = defaultdict(int)

    for palabra, categoria in sinonimos.items():
        if palabra in texto:
            actualizaciones[categoria] += 2  # Subimos a 2 para dar más peso

    for f in features:
        f_base = f.rstrip('s')
        if f in texto or f_base in texto:
            actualizaciones[f] += 2

    return actualizaciones


def contar_gustos_definidos(dic):
    print(dic)
    return sum(v > 0 for v in dic.values())

# Cargar dataset de FAQ
df_faq_origin = pd.read_csv("faq_vina.csv", encoding='latin1', sep=';')
df_faq = df_faq_origin.copy()
df_faq['pregunta_proc'] = df_faq['pregunta'].apply(limpieza)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_faq['pregunta_proc'])

# ---------------------------
# Interfaz de Streamlit
# ---------------------------
st.set_page_config(page_title="Chatbot Turístico", page_icon="🌴")
st.title("👩‍💻 Asistente turístico")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "primer_mensaje" not in st.session_state:
    st.session_state.primer_mensaje = True
if "intereses_usuario" not in st.session_state:
    st.session_state.intereses_usuario = defaultdict(int)

# Mostrar historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# Primer mensaje
if st.session_state.primer_mensaje:
    saludo = "¡Hola! Soy Amukan, tu asistente turístico 🌞 ¿En qué te puedo ayudar hoy?"
    with st.chat_message("assistant"):
        st.markdown(saludo)
    st.session_state.mensajes.append({"role": "assistant", "content": saludo})
    st.session_state.primer_mensaje = False

# Entrada del usuario
if prompt := st.chat_input("Escríbeme lo que buscas..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})

    # Extraer intereses desde keywords
    actualizaciones = extraer_intereses(prompt)
    for k, v in actualizaciones.items():
        st.session_state.intereses_usuario[k] += v

    # Obtener respuesta del chatbot y extraer intereses por categoría
    respuesta, categoria, similitud = responder(prompt)
    with st.chat_message("assistant"):
        st.markdown(respuesta)
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

    # Recomendar si ya hay suficientes gustos definidos
    if contar_gustos_definidos(st.session_state.intereses_usuario) >= 3:
        recomendaciones = recomendar_por_intereses(st.session_state.intereses_usuario)
        mensaje_rec = "🎯 Según tus intereses, te recomiendo:\n\n" + "\n\n".join(recomendaciones)
        with st.chat_message("assistant"):
            st.markdown(mensaje_rec)
        st.session_state.mensajes.append({"role": "assistant", "content": mensaje_rec})

