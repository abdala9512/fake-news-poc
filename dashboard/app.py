import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import openai

from dash_function import get_completion

# MULTIPAGE APP
# https://docs.streamlit.io/library/get-started/multipage-apps/create-a-multipage-app

# Configs
st.set_page_config(layout="wide")

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
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Javeriana.svg/600px-Javeriana.svg.png", width=200)
st.markdown("Esta aplicacion web permite detectar noticias falsas")

st.subheader("Resumen noticia")

# PAG 1 - ANALISIS DESCRIPTIVO DATASET
# verdades vs falsas conteo
# tokens de mayor frecuencia verdades y tokens mayor frecuencia falsas
# Diversidad lexica
# Nube palabras, mayor frecuencia y hapax
# LDA

# PAG 2 = ANALISIS MODELO PREDICTIVO
# NOMBRE Y ESPECIFICACION DEL MODELO, HIPERPARAMETROS
# CURVA ROC
# METRICAS DE CALIDAD
# INTERPRETABILIDAD


# PAG 3 - CLASIFICADOR
# Resumen noticia
# topicos de la noticia
# tabla informativa de la noticia
# resultado modelo



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

# plotly pie chart with random data
left_col_plot.subheader("Distribucion de noticias")
left_col_plot.plotly_chart({
    'data': [
        {
            'labels': ['Fake', 'Real'],
            'values': [10, 20],
            'type': 'pie'

        }
    ],
    'layout': {
        'title': 'Distribucion de noticias falsas y reales'
    }
})

# plotly bar chart with random data
right_col_plot.subheader("Distribucion de noticias")
right_col_plot.plotly_chart({
    'data': [
        {
            'x': ['Fake', 'Real'],
            'y': [10, 20],
            'type': 'bar'
        }
    ],
    'layout': {
        'title': 'Distribucion de noticias falsas y reales'
    }
})



# Model interpretation section
st.subheader("Interpretacion del modelo")
st.markdown("El modelo predice que la noticia es falsa con una probabilidad de 0.8")
st.image("https://shap.readthedocs.io/en/latest/_images/example_notebooks_overviews_An_introduction_to_explainable_AI_with_Shapley_values_37_0.png", width=600)

# Table with random news articles about COVID-19 and their probability of being fake
df = pd.DataFrame({
    'Noticia': ['Noticia 1', 'Noticia 2', 'Noticia 3', 'Noticia 4', 'Noticia 5'],
    'Probabilidad': [0.1, 0.2, 0.3, 0.4, 0.5],
    'Veracidad': ['Fake', 'Real', 'Fake', 'Real', 'Fake']
})
st.subheader("Noticias relacionadas")
st.table(df)

# hover over a word with additional information about the word
st.subheader("Informacion adicional")
st.markdown("""
[id1]: ## "**Aqui** va el significado de una palabra rara Aqui va el significado de una palabra rara Aqui va el significado de una palabra rara Aqui va el significado de una palabra rara"

This is a [**Palabra rara**][id1] example.
""")


# Distribution histogram with random data of length of news articles in words categorized by fake or real
st.subheader("Distribucion de palabras en noticias")

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)


# Group data together
hist_data = [x1, x2]

group_labels = ['Fake', 'Real']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5], )

# Plot!
st.plotly_chart(fig, 
            use_container_width=True,
            layout=dict(title="AHHH"),

    )

# word cloud with random data
left_info_img, right_info_img = st.columns(2)

left_info_img.subheader("Nube de palabras")
left_info_img.image("https://editor.analyticsvidhya.com/uploads/53286wordcloud.PNG", width=600)


right_info_img.subheader("LDA")
right_info_img.image("https://miro.medium.com/v2/resize:fit:1004/1*TRt0p1D-BFZ0WhoR5b_xqw.png", width=600)



import streamlit.components.v1 as components

st.header("test html import")

HtmlFile = open("dashboard/testing.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, height = 1000, width=1300)    

left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(
    f"""
    ### Integrantes Grupo

    ##### Brayan David Rincón Piñeros 
    - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>
    ##### Leonardo Restrepo Alvarez
    - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>
    ##### Giovanni Jimenez Prieto
    - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>
    ##### Miguel Arquez
    - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>

    
    -  Repositorio Gitlab del proyecto: 
    """,
    unsafe_allow_html=True,
)

right_info_col.markdown(
        """
        ### Pontificia Universidad Javeriana

        - Maestría en Analítica para la inteligencia de negocios
         """
    )
