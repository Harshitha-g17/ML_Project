import os
import sys
from dataclasses import dataclass
import pickle 
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Temporarily ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple machine learning models using GridSearchCV and returns a report
    of their accuracy on the test set.
    """
    try:
        report = {}
        best_model_objects = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params.get(model_name, {})

            logging.info(f"Tuning and training {model_name}...")
            print(f"\n--- Starting Grid Search for {model_name} ---") # Added print statement for visibility
            
            if param:
                gs = GridSearchCV(model, param, cv=3, n_jobs=1, verbose=1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            
            report[model_name] = test_model_accuracy
            best_model_objects[model_name] = best_model
            print(f"--- Finished Grid Search for {model_name} ---") # Added print statement for visibility
        
        return report, best_model_objects

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        print(f"ERROR: Error during model evaluation: {e}")
        return None, None


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Splitting training and test input data")
            print("INFO: Splitting training and test input data")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=0),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7, None],
                    "criterion": ["gini", "entropy"]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7]
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [3, 5, 7]
                }
            }

            model_report, best_model_objects = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # MODIFIED: Added a check for model_report being None
            if model_report is None or not model_report:
                logging.error("Model evaluation failed. No report was generated.")
                print("ERROR: Model evaluation failed. No report was generated.")
                return None, None
            
            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]
            best_model_object = best_model_objects[best_model_name]
            
            logging.info(f"Best Model Found: {best_model_name} with Accuracy: {best_score}")
            print(f"INFO: Best Model Found: {best_model_name} with Accuracy: {best_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_object
            )

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)

# --- This is the main block that runs the code ---
if __name__ == "__main__":
    
    np.random.seed(42)
    n_samples = 100 
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    df = pd.DataFrame(data)
    
    df = pd.get_dummies(df, columns=['feature2', 'feature4'], drop_first=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # MODIFIED: stratify=y to ensure both classes are in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    try:
        model_trainer = ModelTrainer()
        best_model_name, best_score = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        
        if best_model_name is not None and best_score is not None:
            print(f"\nModel training process completed. Best model saved: {best_model_name} with score: {best_score}")
        else:
            print("\nModel training failed to complete successfully. Check logs for details.")
            
    except CustomException as e:
        print(f"\nAn error occurred during model training: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")