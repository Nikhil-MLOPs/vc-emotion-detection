import os
import yaml
import pickle
import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# -------------------- Logger Setup --------------------
logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

# -------------------- Helper Functions --------------------
def load_params(path='params.yaml'):
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        model_params = params['model_building']
        logger.info(f"Model parameters loaded: {model_params}")
        return model_params
    except Exception as e:
        logger.error("Failed to load model parameters from YAML.", exc_info=True)
        raise

def load_training_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Training data loaded from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Training data file not found at {path}", exc_info=True)
        raise
    except Exception as e:
        logger.error("Error reading training data.", exc_info=True)
        raise

def train_model(X, y, model_params):
    try:
        model = GradientBoostingClassifier(
            n_estimators=model_params['n_estimators'],
            learning_rate=model_params['learning_rate']
        )
        model.fit(X, y)
        logger.info("Gradient Boosting model training completed.")
        return model
    except Exception as e:
        logger.error("Failed to train Gradient Boosting model.", exc_info=True)
        raise

def save_model(model, path='model.pkl'):
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error("Failed to save the trained model.", exc_info=True)
        raise

# -------------------- Main Pipeline --------------------
def main():
    try:
        logger.info("Model training pipeline started.")

        # Load parameters
        model_params = load_params('params.yaml')

        # Load training data
        train_df = load_training_data('./data/features/train_bow.csv')
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        # Train the model
        model = train_model(X_train, y_train, model_params)

        # Save the trained model
        save_model(model, 'model.pkl')

        logger.info("Model training pipeline completed successfully.")

    except Exception as e:
        logger.critical("Model training pipeline failed.", exc_info=True)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
