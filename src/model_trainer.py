import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from src.logger import get_logger
from src.exception import CustomException

import mlflow
import mlflow.sklearn

import dagshub
dagshub.init(repo_owner='TaufeeqAi', repo_name='colorectal-survival-mlops', mlflow=True)



logger= get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path= "artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok= True)

        logger.info("Model Training Initialized")

    def load_data(self):
        try:
            self.x_train = joblib.load(os.path.join(self.processed_data_path, "x_train.pkl"))
            self.x_test = joblib.load(os.path.join(self.processed_data_path, "x_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Preprocessed data loaded for model training")

        except Exception as e:
            logger.error(f"Error while loading the data for model training: {e}")
            raise CustomException("Failed to load the data model training ")
        
    
    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(self.x_train,self.y_train)

            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))

            logger.info("Model Trained and saved Sucessfully...")
        except Exception as e:
            logger.error(f"Error while Training the model: {e}")
            raise CustomException("Faied Train and save the model")
    
    def evaluate_model(self):
        try:
            y_pred= self.model.predict(self.x_test)
            y_prob = self.model.predict_proba(self.x_test)[ :,1] if len(self.y_test.unique())== 2 else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average= "weighted")
            recall = recall_score(self.y_test, y_pred, average= "weighted")
            f1 = f1_score(self.y_test, y_pred, average= "weighted")

            mlflow.log_metric("Accuracy",accuracy)
            mlflow.log_metric("Precision",precision)
            mlflow.log_metric("Recall",recall)
            mlflow.log_metric("F1-Score",f1)

            logger.info(f"Accuracy: {accuracy}, Precision ={precision}, recall={recall}, f1_score={f1}")

            roc_auc = roc_auc_score(self.y_test,y_prob)
            mlflow.log_metric("ROC-AOC",roc_auc)
            logger.info(f"ROC-AUC score: {roc_auc}")

            logger.info("Model Evaluation Completed...")
        except Exception as e:
            logger.error(f"Error while Evaluating model: {e}")
            raise CustomException("Failed to Evaluate model")

    
    def initiate_model_training_eval(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()


if __name__== "__main__":
    with mlflow.start_run():
        pipeline= ModelTraining()
        pipeline.initiate_model_training_eval()
