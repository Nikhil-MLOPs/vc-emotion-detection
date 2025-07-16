import numpy as np
import pandas as pd

import os
import yaml
import logging

from sklearn.feature_extraction.text import CountVectorizer

# Setup logging configuration
logger = logging.getLogger('feature_engineering')  # Create a logger with name 'data_preprocessing'
logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG

# Create handler for outputting logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Console shows DEBUG and above

# Create handler for writing ERROR logs to a file
file_handler = logging.FileHandler('feature_engineeringn_errors.log')
file_handler.setLevel(logging.ERROR)  # File logs only ERROR and above

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

import os
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Setup logger
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path='params.yaml'):
    """
    Load configuration parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters from {params_path}: {e}")
        raise

def load_data(processed_dir):
    """
    Load preprocessed train and test data.
    """
    try:
        train_path = os.path.join(processed_dir, 'train_processed.csv')
        test_path = os.path.join(processed_dir, 'test_processed.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)

        logger.debug("Processed data loaded and cleaned successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        raise

def vectorize_text(train_texts, test_texts, max_features):
    """
    Apply Bag-of-Words transformation using CountVectorizer.
    """
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(train_texts)
        X_test_bow = vectorizer.transform(test_texts)

        logger.debug("Vectorization (BoW) completed.")
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error(f"Text vectorization failed: {e}")
        raise

def save_features(X_train_bow, y_train, X_test_bow, y_test, output_dir):
    """
    Save feature-engineered data to CSV.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        train_df.to_csv(os.path.join(output_dir, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_bow.csv'), index=False)

        logger.debug("Feature-engineered datasets saved successfully.")
    except Exception as e:
        logger.error(f"Saving feature data failed: {e}")
        raise

def main(params_path='params.yaml'):
    """
    Main function to run feature engineering pipeline.
    """
    try:
        # Load parameters
        params = load_params(params_path)
        max_features = params['feature_engineering']['max_features']

        # Load processed data
        train_df, test_df = load_data('./data/interim')


        # Separate features and labels
        X_train = train_df['content'].values
        y_train = train_df['sentiment'].values
        X_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        # Vectorize content
        X_train_bow, X_test_bow = vectorize_text(X_train, X_test, max_features)

        # Save feature data
        save_features(X_train_bow, y_train, X_test_bow, y_test, './data/features')

    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")

# Entry point
if __name__ == "__main__":
    main()
