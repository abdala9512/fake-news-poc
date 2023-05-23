"""Pipeline - Entrenamiento Modelo Fakenews"""
import mlflow
import pandas as pd
from libs.configs import (
    MLFLOW_TRACKING_URI,
    DATA_FOLDER,
    MLFLOW_FAKE_NEWS_EXPERIMENT_NAME,
)
from typing import Dict, Any, Tuple
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from libs.utils import process_text
from typing import Callable
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

LSTM_SEARCH_SPACE = {
    "MAXLEN": hp.choice("MAXLEN", [100, 200, 300]),
    "EMBED_DIM": hp.choice("EMBED_DIM", [32, 64, 128]),
    "LSTM_SIZE": hp.choice("LSTM_SIZE", [32, 64, 128]),
    "DENSE_SIZE": hp.choice("DENSE_SIZE", [32, 64, 128]),
    "DROPOUT": hp.choice("DROPOUT", [0.1, 0.25, 0.5]),
    "BATCH_SIZE": hp.choice("BATCH_SIZE", [32, 64, 128]),
    "EPOCHS": hp.choice("EPOCHS", [10, 15, 20, 30]),
    "OPTIMIZER": hp.choice("OPTIMIZER", ["adam", "rmsprop"]),
}


def optimize_auc(search_space: Dict, evals: int) -> Callable:
    """decorador para Optimizar hiperametros de cualquier modelo de ML"""

    def _objective_wrapper(objective: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Callable:
            trials = Trials()
            return fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=evals,
                trials=trials,
            )

        return wrapper

    return _objective_wrapper


def define_nn(params: Dict) -> Any:
    """
    Define neural network architecture

    Args:
        params (Dict): neural network parameters

    Returns:
        Any: neural network model
    """
    NeuralNetwork = Sequential()
    NeuralNetwork.add(Input(shape=(params["MAXLEN"],)))
    NeuralNetwork.add(
        Embedding(input_dim=params["VOCAB_SIZE"] + 1, output_dim=params["EMBED_DIM"])
    )
    NeuralNetwork.add(LSTM(params["LSTM_SIZE"]))
    NeuralNetwork.add(Dense(128, activation="relu"))
    NeuralNetwork.add(Dropout(params["DROPOUT"]))
    NeuralNetwork.add(Dense(16, activation="relu"))
    NeuralNetwork.add(Dropout(params["DROPOUT"]))
    NeuralNetwork.add(Dense(1, activation="sigmoid"))
    print("NeuralNetwork architecture: \n")
    print(NeuralNetwork.summary())
    return NeuralNetwork


class PipelineFakeNews:
    """Pipeline entrenamiento Fake news"""

    def __init__(
        self,
        data: pd.DataFrame,
        text_col: str,
        label: str,
        search_space: Dict,
        model_name: str = None,
    ) -> None:
        self.data = data
        self.label = label
        self.text_col = text_col
        self.search_space = search_space
        self.model_name = model_name
        self.processed_data = self._preprocess()

    def _preprocess(self) -> pd.DataFrame:
        """Preprocesamiento de datos"""
        self.data["text_tokenized"] = self.data[self.text_col].apply(
            lambda x: process_text(x, keep_as_list=False)
        )
        return self.data.sample(frac=1)

    def _preprocess_training(self) -> Tuple:
        """Preprocesamiento de datos para entrenamiento

        Returns:
            Tuple: X, y

        """
        tf_tokenizer = Tokenizer()
        fit_text = [" ".join(self.processed_data["text_tokenized"])]
        tf_tokenizer.fit_on_texts(fit_text)
        self.VOCAB_SIZE = len(tf_tokenizer.word_index)

        def text_to_index(text):
            """Convierte un texto a una secuencia de indices"""
            return [tf_tokenizer.word_index[word] for word in text.split(" ")]

        self.processed_data["index_text"] = self.processed_data["text_tokenized"].apply(
            lambda x: text_to_index(x)
        )

        label_binarizer = LabelBinarizer()
        y = label_binarizer.fit_transform(self.processed_data[self.label])
        X = np.array(self.processed_data["index_text"])

        return X, y

    def split_data(self):
        """Split data into train and test"""
        X, y = self._preprocess_training()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=0
        )

    def optimize_models(self, space: Dict) -> Dict:
        """Optimiza hiperparametros de un modelo de ML

        Args:
            space (Dict): espacio de busqueda de hiperparametros

        Returns:
            Dict: hiperparametros optimizados
        """

        @optimize_auc(search_space=space, evals=15)
        def train_predict_lstm(search_space: Dict) -> Dict:

            search_space["VOCAB_SIZE"] = self.VOCAB_SIZE
            self.X_train = tf.keras.preprocessing.sequence.pad_sequences(
                self.X_train, maxlen=search_space["MAXLEN"]
            )
            self.X_test = tf.keras.preprocessing.sequence.pad_sequences(
                self.X_test, maxlen=search_space["MAXLEN"]
            )
            model = define_nn(params=search_space)
            model.compile(
                optimizer=search_space["OPTIMIZER"],
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()],
            )

            model.fit(
                self.X_train,
                self.y_train,
                batch_size=search_space["BATCH_SIZE"],
                epochs=search_space["EPOCHS"],
                validation_data=(self.X_test, self.y_test),
            )
            y_score = model.predict(self.X_test)
            ROC_SCORE = roc_auc_score(self.y_test, y_score)
            print(f"ROC-AUC Score: {ROC_SCORE}")
            return {"loss": -ROC_SCORE, "status": STATUS_OK}

        return train_predict_lstm(space)

    def train_best_model(self, parameters: Dict) -> None:
        """Entrena el mejor modelo

        Args:
            parameters (Dict): hiperparametros del modelo
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_FAKE_NEWS_EXPERIMENT_NAME)
        mlflow.tensorflow.autolog()
        with mlflow.start_run():
            parameters["VOCAB_SIZE"] = self.VOCAB_SIZE
            self.X_train = tf.keras.preprocessing.sequence.pad_sequences(
                self.X_train, maxlen=parameters["MAXLEN"]
            )
            self.X_test = tf.keras.preprocessing.sequence.pad_sequences(
                self.X_test, maxlen=parameters["MAXLEN"]
            )
            nn_model = define_nn(params=parameters)
            nn_model.compile(
                optimizer=parameters["OPTIMIZER"],
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()],
            )

            history = nn_model.fit(
                self.X_train,
                self.y_train,
                batch_size=parameters["BATCH_SIZE"],
                epochs=parameters["EPOCHS"],
                validation_data=(self.X_test, self.y_test),
            )
            _predictions = nn_model.predict(self.X_test)
            mlflow.log_metrics(
                {
                    "precision",
                    precision_score(self.y_test, _predictions),
                    "recall",
                    recall_score(self.y_test, _predictions),
                    "f1",
                    f1_score(self.y_test, _predictions),
                    "accuracy",
                    accuracy_score(self.y_test, _predictions),
                }
            )
            mlflow.log_dict(parameters, "best_params.json")
        mlflow.end_run()

    def run(self):
        # 1. Encontrar mejores hiperparametros
        print("Entrenando mejor modelo y subiendo a MLFlow")
        self.split_data()
        best_params = self.optimize_models(self.search_space)
        best_params = space_eval(self.search_space, best_params)
        print(best_params)
        self.train_best_model(parameters=best_params)


def main():
    """Funcion main para ejecutar pipeline"""
    data = pd.read_csv(f"{DATA_FOLDER}/processed/reconstruidas_newsdf.csv", sep=",")
    # Elegir modelo
    pipeline = PipelineFakeNews(
        data=data, label="Tipo", text_col="Texto", search_space=LSTM_SEARCH_SPACE
    )
    pipeline.run()


# Ejecutar entrenamiento modelo
main()
