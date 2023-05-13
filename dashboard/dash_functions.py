import openai
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd


def get_completion(prompt: str, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )
    return response


plt.rcParams["figure.figsize"] = (20, 10)


def visualize_embedding(
    key_to_vector_embedding: dict, algorithm: Any = TSNE(), word_limit: int = 20
) -> None:
    """Crea una visualizacion de embeddings"""
    assert (
        isinstance(algorithm, PCA)
        or isinstance(algorithm, TSNE)
        or isinstance(algorithm, UMAP)
    ), "La visualizacion solo funciona con instancias PCA, TSNE o UMAP"

    vectors = []
    labels = []
    for key, vector in key_to_vector_embedding.items():
        vectors.append(vector)
        labels.append(key)

    reduced_2d_data = algorithm.fit_transform(np.array(vectors))
    x, y = reduced_2d_data[:, 0], reduced_2d_data[:, 1]

    for i in range(word_limit):
        plt.scatter(x[i], y[i], color="#59C1BD")
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.xlabel("Dim 1", size=15)
    plt.ylabel("Dim 2", size=15)
    plt.title("Representacion Embeddings", size=30)
    plt.show()


def generate_embeddings_2d(
    key_to_vector_embedding: dict, algorithm: Any = TSNE(), word_limit: int = 20
) -> None:
    """Crea una visualizacion de embeddings"""
    assert (
        isinstance(algorithm, PCA)
        or isinstance(algorithm, TSNE)
        or isinstance(algorithm, UMAP)
    ), "La visualizacion solo funciona con instancias PCA, TSNE o UMAP"

    vectors = []
    labels = []
    for key, vector in key_to_vector_embedding.items():
        vectors.append(vector)
        labels.append(key)

    reduced_2d_data = algorithm.fit_transform(np.array(vectors))
    x, y = reduced_2d_data[:, 0], reduced_2d_data[:, 1]

    return pd.DataFrame({"word": labels, "x": x, "y": y}).head(word_limit)
