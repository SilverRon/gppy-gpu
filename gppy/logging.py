import logging
import requests
import sys
from typing import Optional, Union
from . import const


class Logger:
    """
    A flexible logging utility with file, console, and Slack notification capabilities.

    Features:
    - Supports multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Configurable log format
    - Separate debug log file
    - Optional file and console logging
    - Slack notification integration
    """

    def __init__(
        self,
        name: str = "7DT pipeline logger",
        pipeline_name: Optional[str] = None,
        log_file: Optional[str] = None,
        level: str = "INFO",
        log_format: str = "[%(levelname)s] %(asctime)s - %(message)s",
        slack_channel: str = "pipeline_report",
    ):
        self.name = name
        self._log_format = log_format
        self._log_file = log_file
        self._level = level.upper()
        self._pipeline_name = pipeline_name
        self._slack_channel = slack_channel
        self.logger = self._setup_logger()

    def _create_handler(self, handler_type, log_file=None, level=None):
        """Create and configure a log handler."""
        if handler_type == "console":
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(log_file, mode="a")

        handler.setLevel(level or getattr(logging, self._level))
        handler.setFormatter(logging.Formatter(self._log_format))
        return handler

    def _setup_logger(self) -> logging.Logger:
        """Configure and set up the logger with file and console handlers."""
        try:
            log_level = getattr(logging, self._level)
        except AttributeError:
            raise AttributeError(f"Invalid log level: {self._level}")

        # Create logger instance
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to catch all messages

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add console handler
        console_handler = self._create_handler("console", level=log_level)
        logger.addHandler(console_handler)

        # Add file handlers if log_file is specified
        if self._log_file:
            # Main log file with specified level
            file_handler = self._create_handler(
                "file", log_file=self._log_file, level=log_level
            )
            logger.addHandler(file_handler)

            # Debug log file always at DEBUG level
            debug_log_file = self._log_file.replace(".log", "_debug.log")
            debug_handler = self._create_handler(
                "file", log_file=debug_log_file, level=logging.DEBUG
            )
            logger.addHandler(debug_handler)

        return logger

    def log(self, level: Union[int, str], msg: str) -> None:
        """Log a message with a custom log level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.log(level, msg)

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an informational message and potentially send Slack notification."""
        self.logger.info(msg)
        self.send_slack(msg, "INFO")

    def warning(self, msg: str) -> None:
        """Log a warning message and potentially send Slack notification."""
        self.logger.warning(msg)
        self.send_slack(msg, "WARNING")

    def error(self, msg: str) -> None:
        """Log an error message and send Slack notification."""
        self.logger.error(msg)
        self.send_slack(msg, "ERROR")

    def critical(self, msg: str) -> None:
        """Log a critical message and send Slack notification."""
        self.logger.critical(msg)
        self.send_slack(msg, "CRITICAL")

    def send_slack(self, msg: str, level: str) -> None:
        """Send a message to a Slack channel."""
        if not self._pipeline_name:
            return

        msg = f"[{level}, {self._pipeline_name}] {msg}"

        try:
            requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {const.SLACK_TOKEN}"},
                data={"channel": self._slack_channel, "text": msg},
            )
        except Exception as e:
            # Log Slack errors to stderr directly to avoid potential recursion
            print(f"Slack notification failed: {e}", file=sys.__stderr__)

    def set_level(self, level: str) -> None:
        """Update the logging level dynamically."""
        self._level = level.upper()
        log_level = getattr(logging, self._level)
        self.logger.setLevel(log_level)

        # Update non-debug handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if not handler.baseFilename.endswith("_debug.log"):
                    handler.setLevel(log_level)
            else:
                handler.setLevel(log_level)

    def set_pipeline_name(self, name: str) -> None:
        """Set or update the pipeline name for Slack notifications."""
        self._pipeline_name = name

    def set_output_file(self, log_file: str) -> None:
        """Change the log output file and reinitialize logger."""
        self._log_file = log_file
        self.logger = self._setup_logger()

    def set_format(self, fmt: str) -> None:
        """Change the log message format and reinitialize logger."""
        self._log_format = fmt
        self.logger = self._setup_logger()


# Create a singleton logger instance
logger = Logger(name="7DT pipeline logger", slack_channel="pipeline")
