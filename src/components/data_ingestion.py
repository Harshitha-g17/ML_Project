# Importing required libraries
import os                # Provides functions to interact with the operating system (paths, directories, etc.)
import sys               # Provides system-specific parameters and functions (used for exception handling)
from pathlib import Path # For building robust, cross-platform paths

# --- Ensure package imports work even when running this file directly ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exception import CustomException   # Custom exception class for better error handling
from src.logger import logging              # Custom logging utility for logging info/errors
import pandas as pd      # Pandas library for data manipulation and analysis (read CSV, create DataFrames, etc.)

# Scikit-learn function for splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split  
from dataclasses import dataclass   # Decorator for creating simple classes to store configurations

# Importing Data Transformation components (custom code from your project)
try:
    # Use the actual case of the folder (Components) for clarity; Windows is case-insensitive
    from src.Components.data_transformation import DataTransformation  
    from src.Components.data_transformation import DataTransformationConfig  
except (ImportError, AttributeError):
    # Graceful fallback if the module/classes are not implemented yet
    DataTransformation = None  # type: ignore
    DataTransformationConfig = None  # type: ignore

# Importing Model Trainer components (custom code from your project)
try:
    from src.Components.model_trainer import ModelTrainerConfig  
    from src.Components.model_trainer import ModelTrainer  
except (ImportError, AttributeError):
    ModelTrainerConfig = None  # type: ignore
    ModelTrainer = None  # type: ignore

# Configuration class for Data Ingestion (stores file paths for train, test, and raw datasets)
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")   # Path to store training dataset
    test_data_path: str = os.path.join('artifacts', "test.csv")     # Path to store testing dataset
    raw_data_path: str = os.path.join('artifacts', "data.csv")      # Path to store raw dataset

# Main class responsible for reading raw data and splitting into train/test sets
class DataIngestion:
    def __init__(self):
        # Initialize ingestion configuration (paths)
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")   # Log entry into method
        try:
            # Build absolute path to the CSV so it works regardless of current working directory
            data_csv_path = PROJECT_ROOT / 'notebook' / 'data1.csv'
            if not data_csv_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_csv_path}")
            df = pd.read_csv(data_csv_path)
            logging.info('Read the dataset as dataframe')   # Log successful read

            # Create directories if they do not exist (for saving processed data)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset into artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log start of train-test split
            # Split dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training dataset to artifacts/train.csv
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save testing dataset to artifacts/test.csv
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log completion

            # Return paths of train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If error occurs, raise a custom exception with traceback details
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved to: {train_data}")
    print(f"Test data saved to: {test_data}")

    # Only proceed with downstream steps if the stubs are available
    if DataTransformation and ModelTrainer:
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
    else:
        print("DataTransformation / ModelTrainer not implemented yet. Skipping those steps.")