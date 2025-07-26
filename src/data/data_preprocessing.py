import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# --------------------- Logger Setup ---------------------
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:  # Prevent duplicate handlers in notebooks
    logger.addHandler(console_handler)

# --------------------- NLTK Resource Download ---------------------
def download_nltk_resources():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info('NLTK resources "wordnet" and "stopwords" downloaded successfully.')
    except Exception as e:
        logger.error('Failed to download NLTK resources.', exc_info=True)
        raise

# --------------------- Text Preprocessing Functions ---------------------
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized)
    except Exception as e:
        logger.warning('Lemmatization failed for some text.', exc_info=True)
        return text

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in text.split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logger.warning('Stopword removal failed.', exc_info=True)
        return text

def removing_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.warning('Removing numbers failed.', exc_info=True)
        return text

def lower_case(text):
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logger.warning('Lowercasing failed.', exc_info=True)
        return text

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.warning('Punctuation removal failed.', exc_info=True)
        return text

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    except Exception as e:
        logger.warning('URL removal failed.', exc_info=True)
        return text

def remove_small_sentences(df):
    try:
        original_len = len(df)
        df.loc[df['text'].str.split().str.len() < 3, 'text'] = np.nan
        logger.info(f"Removed {original_len - df['text'].count()} short sentences (less than 3 words).")
    except Exception as e:
        logger.warning('Removing small sentences failed.', exc_info=True)

def normalize_text(df, column='content'):
    try:
        logger.info(f"Starting normalization for column: {column}")
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lower_case)
        df[column] = df[column].apply(remove_stop_words)
        df[column] = df[column].apply(removing_numbers)
        df[column] = df[column].apply(removing_punctuations)
        df[column] = df[column].apply(removing_urls)
        df[column] = df[column].apply(lemmatization)
        logger.info("Text normalization completed.")
        return df
    except Exception as e:
        logger.error('Text normalization failed.', exc_info=True)
        raise

# --------------------- Data Loading & Saving ---------------------
def load_data(file_path):
    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading file {file_path}", exc_info=True)
        raise

def save_data(df, path):
    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved processed data to: {path}")
    except Exception as e:
        logger.error(f"Failed to save data to {path}", exc_info=True)
        raise

# --------------------- Main Pipeline ---------------------
def main():
    try:
        logger.info("Pipeline started.")

        # Step 1: Download NLTK resources
        download_nltk_resources()

        # Step 2: Load raw data
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        # Step 3: Normalize text content
        train_processed = normalize_text(train_data, column='content')
        test_processed = normalize_text(test_data, column='content')

        # Step 4: Create output directory
        processed_path = os.path.join("data", "processed")
        os.makedirs(processed_path, exist_ok=True)
        logger.info(f"Directory created or exists: {processed_path}")

        # Step 5: Save processed data
        save_data(train_processed, os.path.join(processed_path, 'train_processed.csv'))
        save_data(test_processed, os.path.join(processed_path, 'test_processed.csv'))

        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.critical("Pipeline failed unexpectedly.", exc_info=True)

# --------------------- Entry Point ---------------------
if __name__ == "__main__":
    main()
