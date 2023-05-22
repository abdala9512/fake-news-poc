"""configurations for the project"""
# PATHS
DATA_FOLDER = "./data/data_fake_news"
PREPROCESSED_DATA = f"{DATA_FOLDER}/preprocessed_data.csv"



# EXPERIMENT TRACKING

MLFLOW_TRACKING_URI="https://dagshub.com/abdala9512/fake-news-poc.mlflow"
MLFLOW_FAKE_NEWS_EXPERIMENT_NAME="fake_news_poc_experiment"
MLFLOW_FAKE_NEWS_MODEL_NAME="fake_news_poc_model"