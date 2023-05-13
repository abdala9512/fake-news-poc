import pandas as pd
import string
from nltk.corpus import stopwords
import re
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

stop_words = set(stopwords.words(["spanish", "english"]))


newsdf = pd.read_csv("data/data_fake_news/processed/processed_data_news.csv", sep="\t")

# Crear una lista de tokens a partir de los textos de noticias
fake_texts = [news_text.split() for news_text in newsdf[newsdf.Tipo == "Falsa"]['Texto']]

# Eliminar las stopwords de cada texto

fake_texts = [[token for token in text if token not in stop_words] for text in fake_texts]
table = str.maketrans('', '', string.punctuation) # Eliminar de los tokens la puntuación parte 1
fake_texts = [[token.translate(table) for token in text] for text in fake_texts] # Eliminar de los tokens la puntuación parte 2
fake_texts = [[re.sub('[¡!¿?\\-»()[]«“”•"<>]“”‘\'’]', '', token) for token in text] for text in fake_texts] # Eliminar caracteres especiales parte 2
fake_texts = [[token.strip() for token in text] for text in fake_texts] # Eliminar tokens vacios parte 1
fake_texts = [[token for token in text if token != ''] for text in fake_texts] # Eliminar tokens vacios parte 2
fake_texts = [[token for token in text if not token.isdigit()] for text in fake_texts] # Eliminar tokens numericos

# Crear el diccionario a partir de los textos
fake_dictionary = corpora.Dictionary(fake_texts)

# Convertir los textos en el formato bag-of-words
corpus = [fake_dictionary.doc2bow(text) for text in fake_texts]

# Entrenar el modelo LDA con el corpus y el diccionario
lda_model_fake = models.ldamodel.LdaModel(corpus=corpus,
                                     id2word=fake_dictionary,
                                     num_topics=5, 
                                     random_state=42,
                                     update_every=1,
                                     chunksize=100,
                                     passes=10,
                                     alpha='auto',
                                     per_word_topics=True)

# Imprimir los tópicos y sus palabras más representativas
for idx, topic in lda_model_fake.print_topics(-1):
    print('Tópico: {} \n Palabras: {}'.format(idx, topic))

# Crear la visualización
vis_data = gensimvis.prepare(lda_model_fake, corpus, fake_dictionary)
pyLDAvis.save_html(vis_data, "dashboard/LDA_fake.html" )

# Crear una lista de tokens a partir de los textos de noticias
true_texts = [news_text.split() for news_text in newsdf[newsdf.Tipo == "Verdadera"]['Texto']]

# Eliminar las stopwords de cada texto

true_texts = [[token for token in text if token not in stop_words] for text in true_texts]
table = str.maketrans('', '', string.punctuation) # Eliminar de los tokens la puntuación parte 1
true_texts = [[token.translate(table) for token in text] for text in true_texts] # Eliminar de los tokens la puntuación parte 2
true_texts = [[re.sub('[¡!¿?\\-»()[]«“”•"<>]“”‘\'’]', '', token) for token in text] for text in true_texts] # Eliminar caracteres especiales parte 2
true_texts = [[token.strip() for token in text] for text in true_texts] # Eliminar tokens vacios parte 1
true_texts = [[token for token in text if token != ''] for text in true_texts] # Eliminar tokens vacios parte 2
true_texts = [[token for token in text if not token.isdigit()] for text in true_texts] # Eliminar tokens numericos

# Crear el diccionario a partir de los textos
true_dictionary = corpora.Dictionary(true_texts)

# Convertir los textos en el formato bag-of-words
corpus = [true_dictionary.doc2bow(text) for text in true_texts]

# Entrenar el modelo LDA con el corpus y el diccionario
lda_model_true = models.ldamodel.LdaModel(corpus=corpus,
                                     id2word=true_dictionary,
                                     num_topics=5, 
                                     random_state=42,
                                     update_every=1,
                                     chunksize=100,
                                     passes=10,
                                     alpha='auto',
                                     per_word_topics=True)

# Imprimir los tópicos y sus palabras más representativas
for idx, topic in lda_model_true.print_topics(-1):
    print('Tópico: {} \n Palabras: {}'.format(idx, topic))

# Crear la visualización
vis_data = gensimvis.prepare(lda_model_true, corpus, true_dictionary)
pyLDAvis.save_html(vis_data, "dashboard/LDA_true.html" )
