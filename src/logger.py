import logging

def get_logger() -> logging.Logger:
    """
    Creates and returns a logger with the name 'dataflow'.
    
    This logger will handle messages with a severity level of INFO or higher.
    
    Returns:
        logging.Logger: A logger object configured for the 'dataflow' namespace.
    """
    # Create or retrieve a logger named 'dataflow'
    logger = logging.getLogger('dataflow')

    # Set the logging level to INFO. This means the logger will handle
    # INFO, WARNING, ERROR, and CRITICAL messages and ignore DEBUG messages.
    logger.setLevel(logging.INFO)

    # If no handlers are already configured for this logger, add a console handler.
    if not logger.handlers:
        # Create a console handler that logs to the standard output (console)
        ch = logging.StreamHandler()

        # Set the logging level for the handler (optional, can be different from the logger's level)
        ch.setLevel(logging.INFO)

        # Create a formatter that determines the format of the log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Set the formatter for the handler
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)
    
    return logger
