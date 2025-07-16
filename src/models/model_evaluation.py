import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Setup logger
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path='model.pkl'):
    """
    Load a trained model from file.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_test_data(test_data_path='./data/features/test_bow.csv'):
    """
    Load test dataset from CSV file.
    """
    try:
        df = pd.read_csv(test_data_path)
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        logger.debug(f"Test data loaded from {test_data_path}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.debug("Model evaluation completed.")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics, output_path='metrics.json'):
    """
    Save evaluation metrics to a JSON file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def main():
    """
    Complete model evaluation pipeline.
    """
    try:
        # Load model and test data
        model = load_model()
        X_test, y_test = load_test_data()

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics)

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")


# Entry point
if __name__ == '__main__':
    main()