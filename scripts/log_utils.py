import logging
import os
from pathlib import Path

def setup_logger(
    name: str = "automl_logger",
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs",
    filename: str = "automl.log"
):
    """
    Set up a reusable logger.
    
    Args:
        name (str): Logger name.
        level (int): Logging level.
        log_to_file (bool): Whether to also log to a file.
        log_dir (str): Directory to save log file if enabled.
        filename (str): Log filename if logging to file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(level)

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        stream_handler.setFormatter(stream_format)
        logger.addHandler(stream_handler)

        # Optional file handler
        if log_to_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_path = Path(log_dir) / filename
            file_handler = logging.FileHandler(file_path)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

    return logger
