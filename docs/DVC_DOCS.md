# Data Version control 

![](https://repository-images.githubusercontent.com/83878269/a5c64400-8fdd-11ea-9851-ec57bc168db5)

## Pasos para versionar datos

### Configurar el servidor remoto de DVC
En nuestro caso es https://dagshub.com/abdala9512/fake-news-poc.dvc, de dagshub.
```
pip install dvc
dvc remote add origin https://dagshub.com/abdala9512/fake-news-poc.dvc
```



```
pip install dagshub
dagshub login

```
### Descargar datos desde Dagshub con DVC
```
dvc pull -r origin
```
### Hacer push de nuevos datos

1. Hacemos un add de los datos
2. subimos la nueva config a github
3. hacemos push de los datos
```
# Add new generated data 
dvc add data/

# commit in git repository
git add data.csv
git commit -m "DVS update msg"
git push origin <branch-name>


# push to DVC
dvc push -r origin data
```

### Para hacer pull de los datos

