import logging
import sys
import os

# Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging settings
log_file_path = os.path.join(log_dir, "project.log")

# Set up general logger to log everything in the project
logging.basicConfig(
    level=logging.INFO,  # You can set this to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode="a")
    ]
)

# Create a general logger for the entire project
logger = logging.getLogger("GeneralLogger")

def get_logger(name=None):
    # Allow specific module logs or fallback to general logger
    return logging.getLogger(name or "GeneralLogger")
