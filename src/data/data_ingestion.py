import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging
## Codes for console handler
logger = logging.getLogger('data_ingestion') # Giving the name of the logger
logger.setLevel('DEBUG') # Setting the level of logger

console_handler = logging.StreamHandler() # Making the terminal handler
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Making the formatter
console_handler.setFormatter(formatter) #Connecting formatter with the handler

logger.addHandler(console_handler) # Adding console handler with the logger finally

## Codes for file handler
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Function to load the test_size parameter from a YAML configuration file
def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:        # Open the YAML file
            params = yaml.safe_load(file)           # Read YAML contents into a Python dictionary
            test_size = params['data_ingestion']['test_size']  # Return the test_size value from the nested structure
        logger.debug("test_size retrieved")
        return test_size
    except FileNotFoundError:
        logger.error("File not found")
        raise Exception(f"YAML file not found at path: {params_path}")
    except yaml.YAMLError as e:
        logger.error("YAML Error")
        raise Exception(f"Error parsing YAML file: {e}")
    except KeyError as e:
        logger.error("Key Error")
        raise Exception(f"Missing key in YAML structure: {e}")

# Function to read the CSV data from a given URL
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)  # Read the CSV file from the given URL using pandas
        logger.debug('Reading the CSV file to DataFrame df')
        return df              # Return the loaded DataFrame
    except Exception as e:
        logger.error("Can't read the DataFrame")
        raise Exception(f"Error reading CSV from URL: {e}")

# Function to process (clean and prepare) the DataFrame
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True) # Drop the 'tweet_id' column since it's not needed for our analysis
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])] # Keep only rows where sentiment is either 'happiness' or 'sadness'
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True) # Replace sentiment values: happiness → 1, sadness → 0 (for binary classification)
        logger.debug('Read the DataFrame and also converted labels')
        return final_df  # Return the cleaned DataFrame
    except KeyError as e:
        logger.error('Column Missing')
        raise Exception(f"Missing expected column in DataFrame: {e}")
    except Exception as e:
        raise Exception(f"Error processing data: {e}")

# Function to save train and test data into CSV files
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)  # Create the directory (e.g., data/raw), if it doesn't exist
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False) # Save the training data as 'train.csv' inside the specified folder
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False) # Save the testing data as 'test.csv' inside the specified folder
        logger.debug('Converted test and train numpy values into CSV')
    except Exception as e:
        raise Exception(f"Error saving data to CSV files: {e}")

# Main function that runs the full pipeline
def main():
    try:
        test_size = load_params('params.yaml') # Step 1: Load test_size parameter from the YAML config file
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv') # Step 2: Read data from the online CSV file
        final_df = process_data(df) # Step 3: Clean and filter the data to keep only happy and sad tweets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42) # Step 4: Split the processed data into training and test sets
        data_path = os.path.join("data", 'raw') # Step 5: Define the path where data should be saved (e.g., data/raw)
        save_data(data_path, train_data, test_data) # Step 6: Save the train and test sets as CSV files in the specified folder
        logger.debug("Called all the functions")
    except Exception as e:
        logger.error("Pipeline can't be made")
        print(f"Pipeline failed due to: {e}")

# This block ensures that main() runs only when this file is executed directly
if __name__ == '__main__':
    main()


