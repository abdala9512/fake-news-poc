# Registro de experimentos

Los experimentos están siendo almacenado en dagshub, con [MLFlow](https://www.mlflow.org/). El proceso para trabajar con experimentos en python es el siguiente:

1. Definir el servidor de MLflow (En nuestro caso el de dagshub)
2. Guardar credenciales en ambiente de trabajo como variables de entorno
3. Tener listo código para entrenar modelos
4. Aplicar logging a los modelos
5. Visualizar resultados en Dagshub

## Servidor de MLflow

Las variables importantes dentro de los pipelines y procesos que tenemos se están guardando en la ruta `libs/configs.py`. Si se quiere obtener alguna variable de ahi solo hay que importar `configs` desde el codigo de python

```python
from libs import configs

print(configs.MLFLOW_TRACKING_URI)
# >>> prints https://dagshub.com/abdala9512/fake-news-poc.mlflow
```

## Credenciales

Las credenciales se pueden extraer de aqui:

![image](https://user-images.githubusercontent.com/42754537/229381618-0a52b16e-bd22-4ebf-873e-f2b7aa154576.png)

Estas se pueden guardar en un archivo `.env` por seguridad

```
MLFLOW_TRACKING_USERNAME=abdala9512
MLFLOW_TRACKING_PASSWORD=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Para obtenerselas dentro de un ambiente de python se puede hacer con `dotenv`

```python
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
```
Y ya estariamos conectados al servidor de **mlflow** con este código:

```python
import mlflow

mlflow.set_tracking_uri(configs.MLFLOW_TRACKING_URI)
```

## Ejemplo Uso MLflow

Antes de empezar, definimos un autologger, que automaticamente buscará las métricas dentro del código y las enviará a MLflow

```python
mlflow.sklearn.autolog()
```

Este código es una clasificador simple que usa un clasificador bayesiano para clasificar noticias falsas
```python

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import auc, accuracy_score, f1_score, recall_score, precision_score, roc_curve

def train_baseline():
    """
    Modelo Baseline Fake news classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(data['news'], data['fake'], random_state = 0)

    clf = GaussianNB()

    with mlflow.start_run():
        
        # model training
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf.fit(X_train_tfidf.toarray(), y_train)
        test_data_transformed = tfidf_transformer.transform(count_vect.transform(X_test)).toarray()
        
        test_predictions = clf.predict(test_data_transformed )
        test_probabilities = clf.predict_proba(test_data_transformed)[:,1]
        
        fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)
        
        accuracy_score_ = accuracy_score(y_test, test_predictions)
        f1_score_ = f1_score(y_test, test_predictions)
        recall_score_ = recall_score(y_test, test_predictions)
        precision_score_ = precision_score(y_test, test_predictions)
        AUC = auc(fpr, tpr)
        
    mlflow.end_run()
```

ejecutamos la función, que entrena el modeloy guarda los resultados en mlflow.

```python
train_baseline()
```

## Visualización de resultados

Ahora podemos analizar y comparar modelos que generamos desde MLflow

![image](https://user-images.githubusercontent.com/42754537/229382148-50634257-022d-45db-a323-7221880a4100.png)

# Estrategias Optimizacion de Hiperparametros