import logging
import sys

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/project.log", mode="a")
    ]
)

def get_logger(name):
    return logging.getLogger(name)