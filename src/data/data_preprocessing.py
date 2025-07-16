# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For handling dataframes
import os  # For file path operations
import re  # For regular expressions
import nltk  # For NLP utilities
import string  # For punctuation constants
from nltk.corpus import stopwords  # For English stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer  # For stemming and lemmatization
import logging  # For logging messages and errors

# Setup logging configuration
logger = logging.getLogger('data_preprocessing')  # Create a logger with name 'data_preprocessing'
logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG

# Create handler for outputting logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Console shows DEBUG and above

# Create handler for writing ERROR logs to a file
file_handler = logging.FileHandler('data_preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)  # File logs only ERROR and above

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Failed to download nltk data: {e}")

# Lemmatization function
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
        text = text.split()  # Split sentence into words
        text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize each word
        return " ".join(text)  # Return sentence after joining lemmatized words
    except Exception as e:
        logger.error(f"Lemmatization failed: {e}")
        return text

# Stopwords removal
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))  # Load English stopwords
        words = [word for word in str(text).split() if word not in stop_words]  # Remove stopwords
        return " ".join(words)
    except Exception as e:
        logger.error(f"Stop word removal failed: {e}")
        return text

# Remove digits from text
def removing_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Removing numbers failed: {e}")
        return text

# Convert text to lowercase
def lower_case(text):
    try:
        words = text.split()  # Split text
        words = [word.lower() for word in words]  # Convert each word to lowercase
        return " ".join(words)
    except Exception as e:
        logger.error(f"Lowercasing failed: {e}")
        return text

# Remove punctuation
def removing_punctuations(text):
    try:
        # Replace punctuation with space
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")  # Remove Arabic semicolon
        text = re.sub('\s+', ' ', text)  # Replace multiple spaces with single space
        return " ".join(text.split()).strip()  # Strip and normalize spaces
    except Exception as e:
        logger.error(f"Removing punctuations failed: {e}")
        return text

# Remove URLs from text
def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)  # Replace URLs with empty string
    except Exception as e:
        logger.error(f"Removing URLs failed: {e}")
        return text

# Remove very short text entries in the DataFrame
def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:  # If text has fewer than 3 words
                df.text.iloc[i] = np.nan  # Replace with NaN
    except Exception as e:
        logger.error(f"Removing small sentences failed: {e}")

# Normalize the content of the dataframe
def normalize_text(df):
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df
    except Exception as e:
        logger.error(f"Text normalization failed: {e}")
        return df

# Main driver function
def main():
    try:
        # Load raw training and testing data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded successfully from raw directory.")
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        return

    try:
        # Apply normalization on train and test datasets
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        return

    try:
        # Define path to save processed data
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)  # Create directory if not exists

        # Save processed datasets to CSV
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

        logger.debug("Processed data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")

# Run the script if this file is executed directly
if __name__ == "__main__":
    main()
