import sys
import os
import pandas as pd
from src.exception import CustomException
from src.util import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 InvoiceNo: str,
                 StockCode: str,
                 Description: str,
                 Quantity: int,
                 InvoiceDate: str,
                 UnitPrice: float,
                 CustomerID: float,
                 Country: str):

        self.InvoiceNo = InvoiceNo
        self.StockCode = StockCode
        self.Description = Description
        self.Quantity = Quantity
        self.InvoiceDate = InvoiceDate
        self.UnitPrice = UnitPrice
        self.CustomerID = CustomerID
        self.Country = Country

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "InvoiceNo": [self.InvoiceNo],
                "StockCode": [self.StockCode],
                "Description": [self.Description],
                "Quantity": [self.Quantity],
                "InvoiceDate": [self.InvoiceDate],
                "UnitPrice": [self.UnitPrice],
                "CustomerID": [self.CustomerID],
                "Country": [self.Country]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
