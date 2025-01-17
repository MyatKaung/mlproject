import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj):
    """
    Saves a Python object (e.g., preprocessor, model) to a file using pickle.
    
    :param file_path: Path to save the object.
    :param obj: The object to save.
    """
    try:
        # Create directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)