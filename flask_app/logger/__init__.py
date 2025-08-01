import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional


class LoggerConfigurator:
    """
    Configures and returns loggers with rotating file and console handlers.

    Attributes:
        log_dir (Path): Directory where log files are stored.
        log_file (Path): Path to the log file.
        max_bytes (int): Maximum size in bytes before rotation.
        backup_count (int): Number of backup files to keep.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        formatter (logging.Formatter): Formatter for log messages.
    """
    def __init__(
        self,
        log_dir: str = 'logs',
        log_file: Optional[str] = None,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        console_level: int = logging.INFO,
        file_level: int = logging.INFO,
        fmt: str = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    ):
        # Determine root directory
        base_dir = Path(__file__).resolve().parents[1]
        self.log_dir = base_dir / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped log file name if not provided
        if not log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{timestamp}.log"
        self.log_file = self.log_dir / log_file

        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = logging.Formatter(fmt)

    def _get_file_handler(self) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            filename=str(self.log_file),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8',
        )
        handler.setLevel(self.file_level)
        handler.setFormatter(self.formatter)
        return handler

    def _get_console_handler(self) -> logging.StreamHandler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.console_level)
        handler.setFormatter(self.formatter)
        return handler

    def configure(
        self,
        name: Optional[str] = None,
        level: int = logging.DEBUG,
    ) -> Logger:
        """
        Creates and configures a logger.

        Args:
            name (Optional[str]): Name of the logger. Root logger if None.
            level (int): Global logging level for the logger.

        Returns:
            Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Prevent duplicate handlers on re-configuration
        if not logger.handlers:
            logger.addHandler(self._get_file_handler())
            logger.addHandler(self._get_console_handler())

        return logger


# Functional helper

def get_logger(
    name: Optional[str] = None,
    log_dir: str = 'logs',
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    fmt: str = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
) -> Logger:
    """
    Helper function to quickly get a configured logger.
    """
    configurator = LoggerConfigurator(
        log_dir=log_dir,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_level=console_level,
        file_level=file_level,
        fmt=fmt,
    )
    return configurator.configure(name)


# Automatically configure root logger on import
_default_configurator = LoggerConfigurator()
_default_configurator.configure()

# Usage example in another module (e.g., src/data/data_ingestion.py):
#
# from src.logger import get_logger
#
# logger = get_logger(__name__)
#
# def ingest_data():
#     logger.info("Starting data ingestion...")
#     try:
#         # ... data ingestion logic ...
#         logger.info("Data ingestion completed successfully.")
#     except Exception as e:
#         logger.exception("Error during data ingestion")
#
# if __name__ == "__main__":
#     ingest_data()
