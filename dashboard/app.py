import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# MULTIPAGE APP
# https://docs.streamlit.io/library/get-started/multipage-apps/create-a-multipage-app


# Configs
st.set_page_config(layout="wide")


st.title("Fake News Detection")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Javeriana.svg/600px-Javeriana.svg.png", width=200)
st.sidebar.title("Fake News Detection")

st.markdown("Esta aplicacion web permite detectar noticias falsas")

st.sidebar.markdown("Esta aplicacion web permite detectar noticias falsas")

st.sidebar.subheader("Ingrese el texto de la noticia")
text = st.sidebar.text_area("Noticia")
st.sidebar.subheader("Ingrese URL de la noticia")
url_text = st.sidebar.text_area("URL")

# checklist in sidebar with the options XGBoost, Logistic Regression and LSTM
st.sidebar.subheader("Seleccione el modelo")
model = st.sidebar.selectbox("Modelo", ("XGBoost", "Logistic Regression", "LSTM"))

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


left_info_col, right_info_col = st.columns(2)

left_info_col.markdown(
    f"""
    ### Authors
    Please feel free to contact us with any issues, comments, or questions.

    ##### Mitchell Parker [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40Mitch_P)](https://twitter.com/Mitch_P)

    - Email:  <mip34@drexel.edu> or <mitchell.parker@fccc.edu>
    - GitHub: https://github.com/mitch-parker

    ##### Roland Dunbrack [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg?style=social&label=Follow%20%40RolandDunbrack)](https://twitter.com/RolandDunbrack)

    - Email: <roland.dunbrack@fccc.edu>
    - GitHub: https://github.com/DunbrackLab
    """,
    unsafe_allow_html=True,
)

right_info_col.markdown(
        """
        ### Funding

        - NIH NIGMS F30 GM142263 (to M.P.)
        - NIH NIGMS R35 GM122517 (to R.D.)
         """
    )

right_info_col.markdown(
    """
    ### License
    Apache License 2.0
    """
)