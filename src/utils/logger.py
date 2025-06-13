import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """Application logger with file and console handlers."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            console: Whether to log to console
            file: Whether to log to file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Add handlers
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        if file:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
        
    def exception(self, message: str) -> None:
        """Log exception message with traceback."""
        self.logger.exception(message)
        
    def set_level(self, level: int) -> None:
        """Set logging level.
        
        Args:
            level: Logging level
        """
        self.logger.setLevel(level)
        
    def get_level(self) -> int:
        """Get current logging level.
        
        Returns:
            Current logging level
        """
        return self.logger.level
        
    def add_handler(self, handler: logging.Handler) -> None:
        """Add a new handler.
        
        Args:
            handler: Logging handler to add
        """
        self.logger.addHandler(handler)
        
    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove a handler.
        
        Args:
            handler: Logging handler to remove
        """
        self.logger.removeHandler(handler)
        
    def get_handlers(self) -> list:
        """Get all handlers.
        
        Returns:
            List of handlers
        """
        return self.logger.handlers 