import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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



def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para = param.get(model_name, {})

            logging.info(f"Evaluating model: {model_name}")
            logging.info(f"Model object: {model}")
            logging.info(f"Hyperparameters: {para}")

            # Check for XGBRegressor and avoid GridSearchCV
            if model_name == "XGBRegressor":
                logging.info(f"Skipping GridSearchCV for {model_name}")
            else:
                if para:
                    logging.info(f"Running GridSearchCV for {model_name}")
                    gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    logging.info(f"Best params for {model_name}: {gs.best_params_}")
                    model.set_params(**gs.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)