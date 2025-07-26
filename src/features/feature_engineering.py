import os
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -------------------- Logging Setup --------------------
logger = logging.getLogger("feature_engineering_bow")
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
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Max features loaded from YAML: {max_features}")
        return max_features
    except Exception as e:
        logger.error("Failed to load parameters from params.yaml", exc_info=True)
        raise

def load_dataset(path):
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace=True)  # Fill missing values
        logger.info(f"Loaded dataset from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {path}", exc_info=True)
        raise

def apply_bow(X_train, X_test, max_features):
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.info(f"Applied CountVectorizer with max_features={max_features}")
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error("Failed during BoW vectorization", exc_info=True)
        raise

def save_dataframe(df, path):
    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved file to: {path}")
    except Exception as e:
        logger.error(f"Could not save file to: {path}", exc_info=True)
        raise

# -------------------- Main Pipeline --------------------
def main():
    try:
        logger.info("BoW feature engineering pipeline started.")

        # Step 1: Load parameters
        max_features = load_params('params.yaml')

        # Step 2: Load processed datasets
        train_data = load_dataset('./data/processed/train_processed.csv')
        test_data = load_dataset('./data/processed/test_processed.csv')

        # Step 3: Extract features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Step 4: Apply Bag of Words
        X_train_bow, X_test_bow = apply_bow(X_train, X_test, max_features)

        # Step 5: Create final dataframes
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        # Step 6: Save to CSV
        output_dir = os.path.join("data", "features")
        os.makedirs(output_dir, exist_ok=True)

        save_dataframe(train_df, os.path.join(output_dir, 'train_bow.csv'))
        save_dataframe(test_df, os.path.join(output_dir, 'test_bow.csv'))

        logger.info("BoW feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.critical("Pipeline execution failed.", exc_info=True)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
