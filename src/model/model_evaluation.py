import os
import json
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# -------------------- Logger Setup --------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

# -------------------- Helper Functions --------------------
def load_model(model_path='model/model.pkl'):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error("Failed to load the model.", exc_info=True)
        raise

def load_test_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Test data loaded from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Test data file not found: {path}", exc_info=True)
        raise
    except Exception as e:
        logger.error("Error loading test data.", exc_info=True)
        raise

def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)

        logger.info("Evaluation metrics calculated successfully.")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
    except Exception as e:
        logger.error("Failed to evaluate the model.", exc_info=True)
        raise

def save_metrics(metrics, path='reports/metrics.json'):
    try:
        with open(path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Evaluation metrics saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {path}", exc_info=True)
        raise

# -------------------- Main Pipeline --------------------
def main():
    try:
        logger.info("Model evaluation pipeline started.")

        # Load model
        model = load_model('model.pkl')

        # Load test data
        test_df = load_test_data('./data/features/test_bow.csv')
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, 'reports/metrics.json')

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.critical("Model evaluation pipeline failed.", exc_info=True)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
