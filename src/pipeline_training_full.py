"""Pipeline - Entrenamiento Modelo Fakenews"""
import mlflow
import pandas as pd
from libs.configs import MLFLOW_TRACKING_URI, DATA_FOLDER
from libs.optimization import optimize_auc, LSTM_SEARCH_SPACE
from typing import Dict, Any
from hyperopt import STATUS_OK
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split


class PipelineFakeNews:
    """Pipeline entrenamiento Fake news"""

    def __init__(
        self,
        data: pd.DataFrame,
        label: str,
        estimator: Any,
        search_space: Dict,
        model_name: str = None,
    ) -> None:
        self.data = data
        self.label = label
        self.estimator = estimator
        self.search_space = search_space
        self.model_name = model_name

    def split_data(self):
        """_summary_"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.drop(self.label, axis=1), self.data[self.label], random_state=0
        )

    def optimize_models(self, space: Dict) -> Dict:
        """_summary_

        Args:
            space (Dict): _description_

        Returns:
            Dict: _description_
        """

        @optimize_auc(search_space=space, evals=10)
        def train_predict_xgboost(search_space: Dict) -> Dict:

            model = self.estimator
            model.fit(self.X_train, self.y_train)
            y_score = model.predict_proba(self.X_test)[:, 1]
            ROC_SCORE = roc_auc_score(self.y_test, y_score)
            print(f"ROC-AUC Score: {ROC_SCORE}")
            return {"loss": -ROC_SCORE, "status": STATUS_OK}

        return train_predict_xgboost(space)

    def train_best_model(self, parameters: Dict):

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.sklearn.autolog()
        with mlflow.start_run():
            self.estimator.fit(self.X_train, self.y_train)
        mlflow.end_run()

    def run(self):
        """_summary_"""
        # 1. Encontrar mejores hiperparametros
        self.split_data()
        best_params = self.optimize_models(self.search_space)
        self.train_best_model(parameters=best_params)


def main():
    """Funcion main para ejecutar pipeline"""
    data = pd.read_csv(f"{DATA_FOLDER}/model_data.csv", sep="\t")
    # Elegir modelo
    model = XGBClassifier(
        n_estimators=int(search_space["n_estimators"])
    )  # Aca podemos cambiar el modelo
    pipeline = PipelineFakeNews(estimator=model, search_space=LSTM_SEARCH_SPACE)
    pipeline.run()


# Ejecutar entrenamiento modelo
main()
