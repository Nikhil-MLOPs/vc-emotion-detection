import os
import pickle
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Set up logger
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_training_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path='params.yaml'):
    """
    Load training hyperparameters from a YAML configuration file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded successfully.")
        return params['model_building']
    except Exception as e:
        logger.error(f"Failed to load parameters from {params_path}: {e}")
        raise


def load_training_data(features_path='./data/features/train_bow.csv'):
    """
    Load training data from CSV and separate into features and labels.
    """
    try:
        df = pd.read_csv(features_path)
        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        logger.debug(f"Training data loaded from {features_path}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


def train_model(X_train, y_train, n_estimators, learning_rate):
    """
    Train a Gradient Boosting Classifier with given parameters.
    """
    try:
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        clf.fit(X_train, y_train)
        logger.debug("Model trained successfully.")
        return clf
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def save_model(model, output_path='model.pkl'):
    """
    Save the trained model using pickle.
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Saving model failed: {e}")
        raise


def main(params_path='params.yaml'):
    """
    Main function to run the training pipeline.
    """
    try:
        # Load parameters
        params = load_params(params_path)

        # Load training data
        X_train, y_train = load_training_data()

        # Train model
        model = train_model(X_train, y_train, params['n_estimators'], params['learning_rate'])

        # Save model
        save_model(model)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")


# Entry point
if __name__ == '__main__':
    main()
