import logging
import sys
import os

def get_logger(script_name=None):
    # Ensure the logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Default to the script name if provided, otherwise use a fallback name
    if not script_name:
        script_name = "default"

    log_file_path = os.path.join(log_dir, f"{script_name}.log")

    # Set up the logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)  # Set the logging level (INFO or DEBUG)

    # Create a file handler that logs to a file in the 'logs' folder
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)  # Log level for the file handler

    # Create a console handler that prints to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter for consistent log formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
