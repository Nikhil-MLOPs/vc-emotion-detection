# Importing required libraries
import numpy as np                          # Importing NumPy for numerical operations (not used in this script directly)
import pandas as pd                         # Importing pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Importing function to split dataset into training and testing sets
import os                                   # Importing os for interacting with the operating system (e.g., file paths)
import yaml                                 # Importing yaml to read configuration files in YAML format
import logging                              # Importing logging module for error and debug logging

# Setting up logger for tracking data ingestion logs
logger = logging.getLogger('data_ingestion')   # Creating a named logger called 'data_ingestion'
logger.setLevel('DEBUG')                       # Setting the logging level to DEBUG to capture all types of logs

# Creating and configuring a handler to print logs to the console
console_handler = logging.StreamHandler()      # Creating a console handler to output logs to the terminal
console_handler.setLevel('DEBUG')              # Setting its level to DEBUG

# Creating and configuring a handler to write error logs to a file
file_handler = logging.FileHandler('errors.log')  # Creating a file handler that logs errors to 'errors.log'
file_handler.setLevel('ERROR')                   # Only log ERROR and above messages to the file

# Defining a log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Formatting log messages
console_handler.setFormatter(formatter)        # Setting the formatter for the console handler
file_handler.setFormatter(formatter)           # Setting the formatter for the file handler

# Adding handlers to the logger
logger.addHandler(console_handler)             # Adding the console handler to the logger
logger.addHandler(file_handler)                # Adding the file handler to the logger

# Function to load parameters (like test_size) from a YAML file
def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:              # Open the specified YAML file
            params = yaml.safe_load(file)                 # Load the YAML contents into a Python dictionary
        test_size = params['data_ingestion']['test_size'] # Extract the test_size value from the nested dictionary
        logger.debug("test_size retrieved")               # Log debug message that test_size was retrieved
        return test_size                                  # Return the extracted test_size
    except Exception as e:
        logger.error(f"Error loading parameters from {params_path}: {e}")  # Log any errors
        raise                                              # Re-raise the exception to halt execution

# Function to read a CSV file from a given URL into a pandas DataFrame
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)               # Read the CSV file from the provided URL into a DataFrame
        return df                           # Return the DataFrame
    except Exception as e:
        logger.error(f"Error reading data from {url}: {e}")  # Log any errors encountered
        raise

# Function to preprocess the DataFrame
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])  # Drop unneeded column

        # Filter and copy to avoid chained assignment issues
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        # Safely replace sentiment labels
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        return final_df
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


# Function to save the train and test DataFrames as CSV files
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')  # Define the path to save raw data
        os.makedirs(raw_data_path, exist_ok=True)                     # Create directory if it doesn't exist
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)  # Save training data to 'train.csv'
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)    # Save test data to 'test.csv'
    except Exception as e:
        logger.error(f"Error saving data to {raw_data_path}: {e}")    # Log any errors during saving
        raise

# Main workflow function
def main():
    try:
        test_size = load_params('params.yaml')                                   # Load test size parameter from YAML file
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')  # Load dataset
        final_df = process_data(df)                                              # Process the dataset
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)  # Split into training and testing sets
        save_data(data_path='./data', train_data=train_data, test_data=test_data)                              # Save train and test data
        print("Data processing complete.")                                       # Print completion message
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")            # Log any unhandled exceptions in main

# Entry point for the script
if __name__ == "__main__":
    main()  # Call the main function if this file is executed directly
