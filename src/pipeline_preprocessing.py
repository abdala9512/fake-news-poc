from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import ConditionalFreqDist
import nltk
# Otras librerias
import pandas as pd
import string
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import chardet
import unidecode
import statistics
import shutil

# import emoji
import os
import codecs
import math

# Analisis de topicos
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


# Stopwords
stop_words = set(stopwords.words(["spanish", "english"]))
others_stop_words = [
    "estan",
    "tambien",
    "si",
    "embargo",
    "aun",
    "traves",
    "ademas",
    "mas",
    "segun",
    "aunque",
    "parte",
    "asi",
]

stop_words.update(others_stop_words)

root_true = "data/data_fake_news/true"
root_fake = "data/data_fake_news/fake"


def rename_files(root, new_path):
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for idx, filename in enumerate(os.listdir(root)):
        if filename.endswith(".txt"):
            new_name = re.sub(r"[^\w\s]", "", filename.split(".")[0])
            new_name = str(idx + 1) + "_" + new_name + ".txt"
            shutil.copy(os.path.join(root, filename), os.path.join(new_path, new_name))


path_fake_rename = "data/data_fake_news/preprocessed/fake_rename"
rename_files(root=root_fake, new_path=path_fake_rename)
path_true_rename = "data/data_fake_news/preprocessed/true_rename"
rename_files(root=root_true, new_path=path_true_rename)

# Noticias falsasfakeCorpus
fakeCorpus = PlaintextCorpusReader(path_fake_rename, ".*")
fake_news = list(set(fakeCorpus.fileids()))

# Carga de archivos
fake_docs_raw = []

# lectura
for file in fake_news:
    with open(path_fake_rename + "/" + file, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result["encoding"]

    try:
        fake_list = PlaintextCorpusReader(path_fake_rename, file, encoding=encoding)
        fake_docs_raw.append(fake_list.raw(file))
    except UnicodeDecodeError:
        print(f"Error: unable to decode {file} with encoding {encoding}")

# cantidad de noticias
print("Cantidad de noticias falsas: ", len(fake_docs_raw))

# Noticias veridicas
trueCorpus = PlaintextCorpusReader(path_true_rename, ".*")
true_news = list(set(trueCorpus.fileids()))

# Carga de archivos
true_docs_raw = []

# lectura
for file in true_news:
    with open(path_true_rename + "/" + file, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result["encoding"]

    try:
        true_list = PlaintextCorpusReader(path_true_rename, file, encoding=encoding)
        true_docs_raw.append(true_list.raw(file))
    except UnicodeDecodeError:
        print(f"Error: unable to decode {file} with encoding {encoding}")

# cantidad de noticias
print("Cantidad de noticias veridicas: ", len(true_docs_raw))


def process_text(textCorpus, nameText, nFQ, nL):
    """
    Esta función permite procesar el texto y permite obtener algunas estadísticas de interes asociadas al texto procesado.
    text: Nombre del texto a procesar.
    nFQ: Número de tokens más frecuentes a obtener.
    nL: logitud para determinar palabras más largas
    """
    # Se obtiene el texto de objeto Corpus
    name = nameText
    raw_text = textCorpus.raw(nameText)

    # 0. Limpieza del texto
    special_chars = [
        "¡",
        "!",
        "¿",
        "?",
        "\\",
        "-",
        "»",
        "[",
        "]",
        "«",
        "•",
        "<",
        ">",
        "(",
        ")",
        "/",
        "&",
        "$",
        '"',
        "“",
        "”",
        "'",
        "‘",
    ]
    raw_text = raw_text.lower()  # minuscula
    raw_text_clean = raw_text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Eliminar puntuación del texto

    for char in special_chars:
        raw_text_clean = raw_text_clean.replace(char, "")
    raw_text_clean_acentos = unidecode.unidecode(raw_text_clean)  # acentos

    # 1. Tokenización
    tokens_doc = nltk.word_tokenize(raw_text_clean.lower())  # Tokenizar el texto

    # 2. Limpieza de tokens

    stop_words = set(stopwords.words(["spanish", "english"]))
    others_stop_words = [
        "estan",
        "tambien",
        "si",
        "embargo",
        "aun",
        "traves",
        "ademas",
        "mas",
        "segun",
        "aunque",
        "parte",
        "asi",
    ]
    stop_words.update(others_stop_words)

    tokens_sw = [
        token for token in tokens_doc if not token in list(stop_words)
    ]  # Seleccionar los tokens que no son stopwords
    table = str.maketrans(
        "", "", string.punctuation
    )  # Eliminar de los tokens la puntuación parte 1
    tokens = [
        w.translate(table) for w in tokens_sw
    ]  # Eliminar de los tokens la puntuación parte 2
    tokens = [
        unidecode.unidecode(token) for token in tokens
    ]  # Eliminar de los tokens los acentos
    tokens = [
        re.sub("[¡!¿?\\-»()[]«“”•\"<>]“”‘'’]", "", token) for token in tokens
    ]  # Eliminar caracteres especiales
    tokens = [token.strip() for token in tokens]  # Eliminar tokens vacios parte 1
    tokens = [
        token for token in tokens if token != ""
    ]  # Eliminar tokens vacios parte 2
    tokens = [
        token for token in tokens if not token.isdigit()
    ]  # Eliminar tokens numericos

    # 3. Cálculo de estadísticas

    # Oraciones y tokens
    num_sent = len(
        textCorpus.sents(nameText)
    )  # Extrae en una lista las oraciones del texto, luego calcula el número de oraciones
    len_news = len(textCorpus.words(nameText)[:])  # logitud del texto

    # Oraciones y tokens promedio
    if len(textCorpus.paras(nameText)) > 0:
        mean_sent_para = sum(len(p) for p in textCorpus.paras(nameText)) / float(
            len(textCorpus.paras(nameText))
        )  # suma(numero oraciones en los parrafos)/(numero de parrafos en el texto)
        mean_tokens_sent = sum(len(s) for s in textCorpus.sents(nameText)) / float(
            len(textCorpus.sents(nameText))
        )  # suma(numero de tokens por oracion)/(numero de oraciones)
    else:
        mean_sent_para = 0
        mean_tokens_sent = 0

    # Número de tokens sin stopwords
    num_tokens_ws = len(
        tokens_sw
    )  # el numero de tokens se calcula sobre el total de tokens en el texto

    # Número de tokens únicos sin stopwords
    num_tokens_unique_ws = len(
        set(tokens_sw)
    )  # el numero tokens unicos se calcula sobre el total de tokens limpios

    # Número de stopwords
    num_stopwords = len_news - len(tokens_sw)  # numero de stopwords

    # Diversidad léxica
    if len(tokens_sw) > 0:
        lexical_diversity = len(set(tokens_sw)) / len(tokens_sw)
    else:
        lexical_diversity = 0

    # Entropia
    token_counts = Counter(tokens_doc)
    token_freqs = [count / len(tokens_doc) for count in token_counts.values()]
    entropy = -sum(freq * math.log2(freq) for freq in token_freqs)

    # 4. tokens de mayor frecuencia
    freq_dist = FreqDist(tokens)
    most_common = freq_dist.most_common(nFQ)
    most_common_tokens = [token[0] for token in most_common]
    num_most_common = len(most_common_tokens)

    # 5. Hapax
    hapaxes = freq_dist.hapaxes()
    num_hapaxes = len(hapaxes)

    # 6. Palabras largas
    long_words = [w for w in freq_dist if len(w) >= nL]
    num_long_word = len(long_words)

    # 7. Referencias a otros sitios
    pattern1 = r"(https?://)?(www\.)?(\w+\.)+[a-zA-Z]{2,}(/\S+)*"
    pattern2 = r"(https?://)?(www\.)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/\S+)*"
    pattern3 = r"(https?://)?(www\.)?([a-zA-Z0-9]+\.)+[a-zA-Z]{2,}(/\S+)*"
    pattern4 = r"(https?://)?(www\.)?([a-zA-Z0-9]+\.)+[a-zA-Z]{2,3}(/\S+)*"
    pattern5 = r"(https?://)?(www\.)?([a-zA-Z]+\.)+[a-zA-Z]{2,3}(/\S+)*"
    pattern6 = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    websites = (
        re.findall(pattern1, raw_text)
        + re.findall(pattern2, raw_text)
        + re.findall(pattern3, raw_text)
        + re.findall(pattern4, raw_text)
        + re.findall(pattern5, raw_text)
        + re.findall(pattern6, raw_text)
    )
    numwebsites = len(websites)

    # # 8. Emojis
    # patron_emoji = re.compile(emoji.get_emoji_regexp())
    # emojis = patron_emoji.findall(raw_text)
    # numemojis = len(emojis) emojis,numemojis,

    return (
        name,
        len_news,
        mean_sent_para,
        mean_tokens_sent,
        num_tokens_ws,
        num_tokens_unique_ws,
        num_stopwords,
        lexical_diversity,
        entropy,
        most_common_tokens,
        num_most_common,
        hapaxes,
        num_hapaxes,
        long_words,
        num_long_word,
        websites,
        numwebsites,
        raw_text_clean_acentos,
    )


tilde = lambda str: str.translate(
    str.maketrans("áàäéèëíìïòóöùúüÀÁÄÈÉËÌÍÏÒÓÖÙÚÜ", "aaaeeeiiiooouuuAAAEEEIIIOOOUUU")
)


def tf_idf_prep(textCorpus, nameText, stemmer=None):
    """
    text: texto a preparar
    stemmer: metodo de stemming, recibe ps = 'PorterStemmer', ss = 'SnowballStemmer'
    """

    name = nameText
    raw_text = textCorpus.raw(nameText)

    # 0. Limpiar texto
    # if isinstance(text, bytes):
    #    text = text.decode('utf-8')

    raw_text = raw_text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Eliminar puntuación
    text = re.sub(
        "[¡!¿?\\-»()[]«“”•]", "", raw_text
    )  # Eliminar algunos caracteres especiales que se identifican en el texto

    # 1. Tokenizar
    tokens = word_tokenize(tilde(text.lower()))

    # 2. Eliminar puntuación
    table = str.maketrans("", "", string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # 3. Segunda limpieza
    tokens = [
        re.sub("“|”|[|¡|!|¿|?|\|\|-|»|(|)|[|]|«|“|”|•|]", "", token) for token in tokens
    ]  # eliminar caracteres especiales
    tokens = [
        token.strip() for token in tokens
    ]  # eliminar espacio en blanco en los tokens
    tokens = [token for token in tokens if token != ""]  # eliminar tokens vacios
    tokens = [
        token for token in tokens if not token.isdigit()
    ]  # eliminar tokens numericos

    # 4. Eliminar stopwords
    stop_words = set(stopwords.words(["spanish", "english"]))
    tokens = [w for w in tokens if not w in stop_words]

    # 5. Stemizar
    if stemmer == "ps":
        stemmer = PorterStemmer()
    elif stemmer == "ss":
        stemmer = SnowballStemmer("spanish")
    else:
        return tokens
    tokens_docs_stem = [stemmer.stem(w) for w in tokens]

    return tokens_docs_stem


fake_data = []

for nameText in fake_news:
    (
        name,
        len_news,
        mean_sent_para,
        mean_tokens_sent,
        num_tokens_ws,
        num_tokens_unique_ws,
        num_stopwords,
        lexical_diversity,
        entropy,
        most_common_tokens,
        num_most_common,
        hapaxes,
        num_hapaxes,
        long_words,
        num_long_word,
        websites,
        numwebsites,
        raw_text_clean_acentos,
    ) = process_text(textCorpus=fakeCorpus, nameText=nameText, nFQ=10, nL=13)
    fake_data.append(
        [
            name,
            len_news,
            mean_sent_para,
            mean_tokens_sent,
            num_tokens_ws,
            num_tokens_unique_ws,
            num_stopwords,
            lexical_diversity,
            entropy,
            most_common_tokens,
            num_most_common,
            hapaxes,
            num_hapaxes,
            long_words,
            num_long_word,
            websites,
            numwebsites,
            raw_text_clean_acentos,
        ]
    )

fakedf = pd.DataFrame(
    fake_data,
    columns=[
        "Archivo",
        "LongitudNoticia",
        "OracionesPromedioParrafo",
        "TokensPromedioOracion",
        "NumTokensWS",
        "NumTokensUnicosWS",
        "NumStopW",
        "DiversidadLexica",
        "Entropia",
        "TokensMayorFrecuencia",
        "NumTokensMayorF",
        "Hapaxes",
        "NumHapaxes",
        "PalabrasLargas",
        "NumPalabrasLargas",
        "WebSites",
        "NumWebSites",
        "Texto",
    ],
)
fakedf["Numero"] = fakedf["Archivo"].str.extract("(\d+)", expand=False).astype(int)
fakedf = fakedf.sort_values(by="Numero", ascending=True)
fakedf["Tipo"] = "Falsa"

true_data = []

for nameText in true_news:
    (
        name,
        len_news,
        mean_sent_para,
        mean_tokens_sent,
        num_tokens_ws,
        num_tokens_unique_ws,
        num_stopwords,
        lexical_diversity,
        entropy,
        most_common_tokens,
        num_most_common,
        hapaxes,
        num_hapaxes,
        long_words,
        num_long_word,
        websites,
        numwebsites,
        raw_text_clean,
    ) = process_text(textCorpus=trueCorpus, nameText=nameText, nFQ=10, nL=13)
    true_data.append(
        [
            name,
            len_news,
            mean_sent_para,
            mean_tokens_sent,
            num_tokens_ws,
            num_tokens_unique_ws,
            num_stopwords,
            lexical_diversity,
            entropy,
            most_common_tokens,
            num_most_common,
            hapaxes,
            num_hapaxes,
            long_words,
            num_long_word,
            websites,
            numwebsites,
            raw_text_clean,
        ]
    )

truedf = pd.DataFrame(
    true_data,
    columns=[
        "Archivo",
        "LongitudNoticia",
        "OracionesPromedioParrafo",
        "TokensPromedioOracion",
        "NumTokensWS",
        "NumTokensUnicosWS",
        "NumStopW",
        "DiversidadLexica",
        "Entropia",
        "TokensMayorFrecuencia",
        "NumTokensMayorF",
        "Hapaxes",
        "NumHapaxes",
        "PalabrasLargas",
        "NumPalabrasLargas",
        "WebSites",
        "NumWebSites",
        "Texto",
    ],
)

truedf["Numero"] = truedf["Archivo"].str.extract("(\d+)", expand=False).astype(int)
truedf = truedf.sort_values(by="Numero", ascending=True)
truedf["Tipo"] = "Verdadera"

newsdf = pd.concat(
    [fakedf, truedf], ignore_index=True
)  # generate_categorical_boxplot(df, categorical_col, numerical_col, title, xlabel, ylabel)
newsdf = newsdf[newsdf.LongitudNoticia > 0]
newsdf["Key"] = newsdf["Tipo"].str[0] + newsdf["Numero"].astype(str)

newsdf.to_csv("data/data_fake_news/processed/processed_data_news.csv", index=False, sep="\t")
print("DF generado.")
