import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

features = ["Entropia", "DiversidadLexica", "LongitudNoticia", "TokensPromedioOracion",
            "NumTokensWS", "NumTokensUnicosWS", "NumHapaxes", "NumPalabrasLargas"]
target = "Tipo"

newsdf = pd.read_csv("dashboard/dash_data/processed_data_news.csv", sep="\t")

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


# Preparación de los datos
X = newsdf['Texto']
y = newsdf['Tipo']

# Codifica la variable objetivo como 0 y 1
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Tokeniza y crea secuencias de texto
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

# Ajusta la longitud de las secuencias
max_length = max([len(s) for s in sequences])
data = pad_sequences(sequences, maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)


# Ajuste del modelo
embedding_dim = 100
vocab_size = len(word_index) + 1
lstm_units = 64

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(lstm_units),
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)
