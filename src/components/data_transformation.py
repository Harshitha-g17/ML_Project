import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def _init_(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["Quantity", "UnitPrice"]
            categorical_columns = [
                "Country",
                "StockCode",
                "Description"
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Basic data cleaning: drop rows where CustomerID or Description are missing, 
            # as these are critical for the dataset's integrity even in a raw approach.
            train_df.dropna(subset=['CustomerID', 'Description'], inplace=True)
            test_df.dropna(subset=['CustomerID', 'Description'], inplace=True)
            
            

            # Calculate 'Total_Price' if it's not already present in the DataFrame.
            # This will be our target variable.
            if 'Total_Price' not in train_df.columns:
                train_df['Total_Price'] = train_df['Quantity'] * train_df['UnitPrice']
            if 'Total_Price' not in test_df.columns:
                test_df['Total_Price'] = test_df['Quantity'] * test_df['UnitPrice']

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            # The target column for this raw data approach is the 'Total_Price' of each transaction.
            target_column_name="Total_Price"
            
            # Columns to be dropped from the input features (e.g., identifiers that are not features)
            # Ensure only columns present in the DataFrame are attempted to be dropped.
            columns_to_drop_from_features = [target_column_name, 'InvoiceNo', 'CustomerID', 'InvoiceDate']
            
            input_feature_train_df = train_df.drop(
                columns=[col for col in columns_to_drop_from_features if col in train_df.columns], 
                axis=1
            )
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[col for col in columns_to_drop_from_features if col in test_df.columns], 
                axis=1
            )
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the preprocessing pipelines (fit on train, transform on both)
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Concatenate the transformed features with the target variable for model training
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Ensure the 'artifacts' directory exists before attempting to save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)