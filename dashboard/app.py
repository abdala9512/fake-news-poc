import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import streamlit.components.v1 as components
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pickle
from dash_functions import generate_embeddings_2d

# Configs
st.set_page_config(layout="wide")
stop_words = set(stopwords.words(["spanish", "english"]))


# Datos y modelos

COLUMNS_TO_DROP = ["Texto", "Hapaxes", "PalabrasLargas"]
data = pd.read_csv("dashboard/dash_data/processed_data_news.csv", sep="\t")
# Embeddings
#with open("embeddings_news.pickle", "rb") as handle:
#    embeddings_vector = pickle.load(handle)


# import openai

# from dash_function import get_completion

# MULTIPAGE APP
# https://docs.streamlit.io/library/get-started/multipage-apps/create-a-multipage-app


# SIDE BAR ---------------------------------------------------------

st.sidebar.markdown("Esta aplicacion web permite detectar noticias falsas")
st.sidebar.subheader("Ingrese el texto de la noticia")
st.sidebar.title("Fake News Detection")
text = st.sidebar.text_area("Noticia")
st.sidebar.subheader("Ingrese URL de la noticia")
url_text = st.sidebar.text_area("URL")

# checklist in sidebar with the options XGBoost, Logistic Regression and LSTM
st.sidebar.subheader("Seleccione el modelo")
model = st.sidebar.selectbox("Modelo", ("XGBoost", "Logistic Regression", "LSTM"))


# PAG PRINCIPAL ---------------------------------------------------------------

st.title("Proyecto de NLP para la Identificación de Noticias Falsas Acerca de COVID-19")
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Javeriana.svg/600px-Javeriana.svg.png",
    width=200,
)
st.markdown(
    """
Esta aplicacion web permite detectar noticias falsas y tiene uso como herramienta
Pedagogica para la identificacion de noticias falsas e interpretacion de resultados.

Los componente de la aplicacion son:
- **Analisis descriptivo**: El analisis descriptivo permite entender la distribucion de las noticias y los topicos de las noticias.
- **ingenieria de caracteristicas**: La ingenieria de caracteristicas permite entender como se transforman las noticias en vectores.
- **Modelo predictivo**: El modelo predictivo permite predecir si una noticia es falsa o verdadera.
- **Interpretacion del modelo**: El modelo predictivo es interpretable y permite entender como se toman las decisiones.
- **Clasificador**: El clasificador permite ingresar una noticia y obtener un resumen de la noticia, los topicos de la noticia y la veracidad de la noticia.

"""
)

st.subheader("Analisis Descriptivo")
st.markdown(
    """
En esta seccion se presenta el analisis descriptivo de las noticias. 
El dataset esta compuesto por 1000 noticias falsas y 1000 noticias verdaderas extraidas de medios de comunicacion en internet.

- **Distribucion de noticias**: La distribucion de noticias permite entender la distribucion de noticias falsas y verdaderas.
- **Distribucion de palabras**: La distribucion de palabras permite entender la distribucion de palabras en las noticias falsas y verdaderas.
- **Distribucion de topicos**: La distribucion de topicos permite entender la distribucion de topicos en las noticias falsas y verdaderas.
"""
)


st.table(data.drop(COLUMNS_TO_DROP, axis=1).head(1))


# try:
#     response = get_completion(
#         prompt=f""""
#             Genera un resumen de 50 palabras de la siguiente noticia {text}
#         """
#     )
# except:
#     pass
# st.markdown(response.to_dict()["choices"][0]["message"]["content"])


# two columns for the plotly charts
left_col_plot, right_col_plot = st.columns(2)


# plotly bar chart with random data
left_col_plot.subheader("Distribucion de noticias")
news_count_fig = px.pie(
    data.groupby("Tipo")[["Tipo"]]
    .count()
    .rename(columns={"Tipo": "Num"})
    .reset_index(),
    values="Num",
    names="Tipo",
    title="Distribucion de noticias falsas y reales",
)
left_col_plot.plotly_chart(news_count_fig, theme="streamlit", use_container_width=True)


right_col_plot.subheader("Distribucion de Diversidad Lexica")
boxplot_diversity = px.box(data, y="DiversidadLexica", x="Tipo")

right_col_plot.plotly_chart(boxplot_diversity, use_container_width=True)
# Distribution histogram with random data of length of news articles in words categorized by fake or real
st.subheader("Distribucion del numero de palabras en noticias")

# Create distplot with custom bin_size
fig = px.histogram(
    data, x="LongitudNoticia", color="Tipo", marginal="rug", hover_data=data.columns
)

# Plot!
st.plotly_chart(
    fig,
    use_container_width=True,
    layout=dict(title="AHHH"),
)


st.subheader("Analisis Nubes de palabtas")

wordcloud_keys = {
    "Hapaxes": {
        "column": "Hapaxes",
        "explanation": "Las palabras hapaxes son palabras que aparecen una sola vez en el corpus",
    },
    "Palabras totales": {
        "column": "Texto",
        "explanation": "Las palabras totales son todas las palabras que aparecen en el corpus",
    },
    "Palabras mas frecuentes": {
        "column": "TokensMayorFrecuencia",
        "explanation": "Las palabras mas frecuentes son las palabras que aparecen mas de 10 veces en el corpus",
    },
    "Palabras mas largas": {
        "column": "PalabrasLargas",
        "explanation": "Las palabras mas largas son las palabras que tienen mas de 10 caracteres",
    },
}

wordcloud_analysis = st.selectbox(
    "Seleccione el analisis",
    ("Palabras totales", "Hapaxes", "Palabras mas frecuentes", "Palabras mas largas"),
)

st.markdown(
    f"""Analisis de **{wordcloud_analysis}**: {wordcloud_keys[wordcloud_analysis]["explanation"]}
"""
)

# word cloud with random data
left_info_img, right_info_img = st.columns(2)


fake_news = " ".join(
    data[data["Tipo"] == "Falsa"][wordcloud_keys[wordcloud_analysis]["column"]].tolist()
)
true_news = " ".join(
    data[data["Tipo"] == "Verdadera"][
        wordcloud_keys[wordcloud_analysis]["column"]
    ].tolist()
)

word_cloud_fake = WordCloud(
    width=800,
    height=800,
    background_color="white",
    stopwords=stop_words,
    min_font_size=10,
).generate(fake_news)

word_cloud_true = WordCloud(
    width=800,
    height=800,
    background_color="white",
    stopwords=stop_words,
    min_font_size=10,
).generate(true_news)

left_info_img.subheader("Noticias Verdaderas")
plt.imshow(word_cloud_true, interpolation="bilinear")
plt.axis("off")
left_info_img.image(word_cloud_true.to_array())


right_info_img.subheader("Noticias Falsas")
plt.imshow(word_cloud_true, interpolation="bilinear")
plt.axis("off")
right_info_img.image(word_cloud_fake.to_array())


st.header("LDA - Latent Dirichlet Allocation - Creacion de topicos")
st.markdown(
    """

[lda]: ## "LDA es un modelo estadístico que se utiliza para identificar temas en grandes conjuntos de datos, como textos. Es una herramienta útil para entender las principales ideas y patrones en un conjunto de documentos, permitiendo que los usuarios clasifiquen y analicen la información de manera más eficiente."

En esta seccion se presenta la creacion de topicos.
Para la identificacion de topicos se utilizo el algoritmo [**LDA - Latent Dirichlet Allocation**][lda] con el
objetivo de identificar los topicos de las noticias. Se usa la libreria [**pyLDAvis**](https://github.com/bmabey/pyLDAvis/) para la visualizacion interactiva.
"""
)
lda_select = st.selectbox("Seleccione el numero de topicos", ("LDA Falsas", "LDA Verdaderas"))

lda_dict = {
    "LDA Falsas": "dashboard/LDA_fake.html",
    "LDA Verdaderas": "dashboard/LDA_true.html",
}

HtmlFile = open(lda_dict[lda_select], "r", encoding="utf-8")
source_code = HtmlFile.read()
components.html(source_code, height=1000, width=1300)

st.header("Word Embedding - Representacion de palabras en vectores")
st.markdown(
    """
[embedding]: ## "Word Embedding es una tecnica de procesamiento de lenguaje natural que permite representar palabras como vectores de numeros reales. Esta tecnica permite representar palabras en un espacio vectorial donde palabras similares estan cerca entre si. Esta tecnica permite representar palabras en un espacio vectorial donde palabras similares estan cerca entre si."

En esta seccion se presenta la representacion de palabras en vectores con [word embeddings][embedding].
"""
)

# embeddings_data = generate_embeddings_2d(
#     key_to_vector_embedding=embeddings_vector, word_limit=100
# )
# embeddings_chart = px.scatter(embeddings_data, x="x", y="y", text="word", size_max=60)
# embeddings_chart.update_traces(textposition="top right")
# st.plotly_chart(
#     embeddings_chart, use_container_width=True, layout=dict(title="Word Embeddings")
# )
# Model interpretation section
st.subheader("Analisis Modelos Predictivos")

st.markdown(
    """

[roc]: ## "La curva ROC y el AUC son medidas que se utilizan para evaluar la capacidad predictiva de un modelo en problemas binarios. La curva ROC muestra la relación entre la tasa de verdaderos positivos y la tasa de falsos positivos, mientras que el AUC es una medida del rendimiento general del modelo. En resumen, son herramientas para evaluar qué tan bien un modelo puede distinguir entre dos clases."
En esta seccion se presenta el analisis del modelo predictivo.
Los modelos entrenados son:
- **XGBoost**: El modelo XGBoost es un modelo de arboles de decision.
- **Logistic Regression**: El modelo Logistic Regression es un modelo de regresion logistica.
- **LSTM**: El modelo LSTM es un modelo de redes neuronales recurrentes.

Los componentes del analisis son:
- **[Curva ROC][roc]**: La curva ROC permite entender la calidad del modelo predictivo.
- **Metricas de calidad**: Las metricas de calidad permiten entender la calidad del modelo predictivo.
- **Interpretacion del modelo**: La interpretacion del modelo permite entender como se toman las decisiones del modelo predictivo.
"""
)

st.markdown("El modelo predice que la noticia es falsa con una probabilidad de 0.8")
st.image(
    "https://shap.readthedocs.io/en/latest/_images/example_notebooks_overviews_An_introduction_to_explainable_AI_with_Shapley_values_37_0.png",
    width=600,
)

# Table with random news articles about COVID-19 and their probability of being fake
df = pd.DataFrame(
    {
        "Noticia": ["Noticia 1", "Noticia 2", "Noticia 3", "Noticia 4", "Noticia 5"],
        "Probabilidad": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Veracidad": ["Fake", "Real", "Fake", "Real", "Fake"],
    }
)
st.subheader("Noticias relacionadas")
st.table(df)

st.subheader("Clasificador")

st.markdown(
    """
En esta seccion se presenta el clasificador de noticias falsas y verdaderas.

- **Resumen de la noticia**: El resumen de la noticia permite entender el contenido de la noticia.
- **Topicos de la noticia**: Los topicos de la noticia permiten entender los topicos de la noticia.
- **Tabla informativa**: La tabla informativa permite entender la veracidad de la noticia.
- **prediction**: La prediccion permite entender la veracidad de la noticia.
"""
)


left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(
    f"""
    ### Integrantes Grupo

    ##### Brayan David Rincón Piñeros 
    - Email:  <bdavid_rincon@javeriana.edu.co> 
    ##### Leonardo Restrepo Alvarez
    - Email:  <le.restrepo@javeriana.edu.co> 
    ##### Giovanni Jimenez Prieto
    - Email:  <jimenezgiovanni@javeriana.edu.co> 
    ##### Miguel Arquez
    - Email:  <arquez.m@javeriana.edu.co> 

    
    -  Repositorio Gitlab del proyecto: 
    """,
    unsafe_allow_html=True,
)

right_info_col.markdown(
    """
        ### Pontificia Universidad Javeriana

        - Maestría en Analítica para la inteligencia de negocios, 2023
         """
)
