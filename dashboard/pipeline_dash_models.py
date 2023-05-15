import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from dash_functions import process_text, text_to_index, MAX_LEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

features = ["Entropia", "DiversidadLexica", "LongitudNoticia", "TokensPromedioOracion",
            "NumTokensWS", "NumTokensUnicosWS", "NumHapaxes", "NumPalabrasLargas"]
target = "Tipo"

newsdf = pd.read_csv("dashboard/dash_data/processed_data_news.csv", sep="\t")
newsdf["text_tokenized_list"] = newsdf["Texto"].apply(lambda x: process_text(x, keep_as_list=True))
newsdf["text_tokenized"] = newsdf["Texto"].apply(lambda x: process_text(x, keep_as_list=False))
newsdf = newsdf.sample(frac = 1)

# REGRESTION LOGISTICA

X = newsdf[features]
y = newsdf[target].apply(lambda x: 1 if x == "Verdadera" else 0)  # Codifica la variable objetivo como 1 y 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste del modelo y resumen estadístico
X_train = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary())

# NN\



label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(newsdf["Tipo"])
tf_tokenizer = Tokenizer()
fit_text = [" ".join(newsdf["text_tokenized"])]
tf_tokenizer.fit_on_texts(fit_text)

MAX_LEN = 100
VOCAB_SIZE = len(tf_tokenizer.word_index)
EMBED_DIM = 100

newsdf["index_text"] = newsdf["text_tokenized"].apply(lambda x: text_to_index(x))

X = np.array(newsdf["index_text"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_LEN)

def define_nn():
    NeuralNetwork = Sequential()
    NeuralNetwork.add(Input(shape=(MAX_LEN,)))
    NeuralNetwork.add(Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBED_DIM))
    NeuralNetwork.add(LSTM(128))
    NeuralNetwork.add(Dense(128, activation="relu"))
    NeuralNetwork.add(Dropout(0.1))
    NeuralNetwork.add(Dense(16, activation="relu"))
    NeuralNetwork.add(Dropout(0.1))
    NeuralNetwork.add(Dense(1, activation="sigmoid"))
    print('NeuralNetwork architecture: \n')
    print(NeuralNetwork.summary())  
    return NeuralNetwork

nn_model = define_nn()
nn_model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])



history = nn_model.fit(X_train, y_train, 
                    batch_size=64, epochs=20, 
                    validation_data=(X_test, y_test)
                   )

# Save models

MODELS ={
    'logistic_regression': result,
    'neural_network': nn_model
}

for _model in MODELS.keys():
    
    if _model == 'logistic_regression':
        with open(f'dashboard/dash_data/{_model}.pkl','wb') as f:
            pickle.dump(MODELS[_model],f)
    else:
        MODELS[_model].save(f'dashboard/dash_data/{_model}')
