from src.logger import get_logger
from src.exception import CustomException
import sys
logger=get_logger(__name__)
def divide_number(a,b):
    try:
        result=a/b
        logger.info(f"Division of 2 numbers")
        return result
    except Exception as e:
        logger.error("An unexpected error occurred during division.")
        raise CustomException("Custom Error Zero", sys)
if __name__=="__main__":
    try:
        logger.info("Starting the division operation")
        divide_number(10,2)
    except CustomException as ce:
        logger.error(str(ce)) 
    