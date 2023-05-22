# Despliegue solucion


Para ejecutar el pipeline se deben seguir los siguientes pasos

1. configurar ambiente virtual

```
pipenv install
```

2. Activar ambiente virtual
```
pipenv shell
```

3. Configurar archivo `.env`

```
MLFLOW_TRACKING_USERNAME=abdala9512
MLFLOW_TRACKING_PASSWORD=********************************
```

4. Ejecutar `pipeline_training_full.py

```
python src/pipeline_training_full.py
```