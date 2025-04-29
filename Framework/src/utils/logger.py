import logging

def setup_logger():
    """
    Sets up and returns a logger with a standard format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('medicinal-chatbot')
    
    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
