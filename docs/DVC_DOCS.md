# Data Version control 

![](https://repository-images.githubusercontent.com/83878269/a5c64400-8fdd-11ea-9851-ec57bc168db5)

## Data versioning steps

### Set DVC remote
```
dvc remote add origin https://dagshub.com/abdala9512/fake-news-poc.dvc
```

### Push new data
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