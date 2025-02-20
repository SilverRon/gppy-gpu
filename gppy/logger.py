import logging
import requests
import sys
import time
from typing import Optional, Union, Dict, Any
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

    Attributes:
    - name (str): The name of the logger instance.
    - pipeline_name (str): The name of the pipeline for Slack notifications.
    - log_file (str): The file path for log output.
    - level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    - log_format (str): The format of log messages.
    - slack_channel (str): The Slack channel for notifications.

    Methods:
    - log: Log a message with a custom log level.
    - debug: Log a debug message.
    - info: Log an informational message and potentially send Slack notification.
    - warning: Log a warning message and potentially send Slack notification.
    - error: Log an error message and send Slack notification.
    - critical: Log a critical message and send Slack notification.
    - send_slack: Send a message to a Slack channel, ensuring all follow-up messages stay in a thread.
    - set_level: Update the logging level dynamically.
    - set_pipeline_name: Set or update the pipeline name for Slack notifications.
    - set_output_file: Change the log output file and reinitialize logger.
    - set_format: Change the log message format and reinitialize logger.
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
        self._name = name
        self._log_format = log_format
        self._log_file = log_file
        self._level = level.upper()
        self._pipeline_name = pipeline_name
        self._slack_channel = slack_channel
        self._thread_ts = None  # Store the first message timestamp
        self.logger = self._setup_logger()
        
        # Redirect stdout and stderr to the logger
        sys.stdout = StdoutToLogger(self.logger)
        sys.stderr = StderrToLogger(self.logger)

    def _create_handler(self, handler_type, log_file=None, level=None, mode="a"):
        """
        Create and configure a log handler for different output streams.

        This method supports creating handlers for console (stdout/stderr) 
        and file-based logging with configurable levels and formats.

        Args:
            handler_type (str): Type of handler ('console', 'console_err', or 'file')
            log_file (str, optional): Path to the log file for file handlers
            level (int, optional): Logging level for the handler
            mode (str, optional): File writing mode, defaults to append ('a')

        Returns:
            logging.Handler: Configured log handler

        Raises:
            IOError: If there are issues creating console or file handlers
        """
        if handler_type == "console":
            try:
                handler = logging.StreamHandler(sys.__stdout__)
            except (IOError, OSError) as e:
                print(f"Error creating stdout handler: {e}", file=sys.__stderr__)
                handler = logging.StreamHandler()
        elif handler_type == "console_err":
            try:
                handler = logging.StreamHandler(sys.__stderr__)
            except (IOError, OSError) as e:
                print(f"Error creating stderr handler: {e}", file=sys.__stderr__)
                handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(log_file, mode=mode)

        handler.setLevel(level or getattr(logging, self._level))
        handler.setFormatter(logging.Formatter(self._log_format))
        return handler

    def _setup_logger(self, overwrite:bool=True) -> logging.Logger:
        """
        Configure and set up the logger with multiple handlers.

        This method initializes the logging system with:
        - Console handlers for standard output and error
        - File handlers for logging to files (if specified)
        - Configurable log levels and formats

        Returns:
            logging.Logger: Fully configured logger instance

        Raises:
            AttributeError: If an invalid log level is provided
        """
        try:
            log_level = getattr(logging, self._level)
        except AttributeError:
            raise AttributeError(f"Invalid log level: {self._level}")

        # Create logger instance
        logger = logging.getLogger(self._name)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to catch all messages

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add console handler
        console_handler = self._create_handler("console", level=logging.INFO)
        logger.addHandler(console_handler)
        console_err_handler = self._create_handler("console_err", level=logging.ERROR)
        logger.addHandler(console_err_handler)

        # Add file handlers if log_file is specified
        if self._log_file:
            # Main log file with specified level
            file_handler = self._create_handler(
                "file", log_file=self._log_file, level=log_level, mode="w" if overwrite else "a"
            )
            logger.addHandler(file_handler)

            # Debug log file always at DEBUG level
            debug_log_file = self._log_file.replace(".log", "_debug.log")
            debug_handler = self._create_handler(
                "file", log_file=debug_log_file, level=logging.DEBUG
            )
            logger.addHandler(debug_handler)

        return logger
        
    def log(self, level: Union[int, str], msg: str, **kwargs) -> None:
        """
        Log a message with a custom log level.

        Allows logging with both numeric and string-based log levels.

        Args:
            level (Union[int, str]): Logging level (e.g., logging.INFO or 'INFO')
            msg (str): Message to log
            **kwargs: Additional keyword arguments for logging
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.log(level, msg, **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        """
        Log a debug message.

        Args:
            msg (str): Debug message to log
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        """
        Log an informational message and send a Slack notification.

        Args:
            msg (str): Informational message to log
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.info(msg, **kwargs)
        self.send_slack(msg, "INFO")

    def warning(self, msg: str, **kwargs) -> None:
        """
        Log a warning message and send a Slack notification.

        Args:
            msg (str): Warning message to log
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.warning(msg, **kwargs)
        self.send_slack(msg, "WARNING")

    def error(self, msg: str, **kwargs) -> None:
        """
        Log an error message and send a Slack notification.

        Args:
            msg (str): Error message to log
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.error(msg, **kwargs)
        self.send_slack(msg, "ERROR")

    def critical(self, msg: str, **kwargs) -> None:
        """
        Log a critical message and send a Slack notification.

        Args:
            msg (str): Critical message to log
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.critical(msg, **kwargs)
        self.send_slack(msg, "CRITICAL")

    def send_slack(self, msg: str, level: str) -> None:
        """
        Send a message to a Slack channel with thread support.

        Sends log messages to a specified Slack channel, maintaining 
        a single thread for related messages. Handles potential 
        communication errors gracefully.

        Args:
            msg (str): Message to send to Slack
            level (str): Log level of the message (INFO, WARNING, etc.)
        """
        if not self._pipeline_name:
            return

        msg = f"[`{level}`] {msg}"

        try:
            payload = {"channel": self._slack_channel, "text": msg}

            # Use thread_ts if available to continue the thread
            if self._thread_ts:
                payload["thread_ts"] = self._thread_ts

            response_data = self._send_slack_with_retry(payload)

            if response_data is None:
                return

            # If this is the first message, store the thread_ts for replies
            if not self._thread_ts and response_data.get("ok"):
                self._thread_ts = response_data["ts"]

            if not response_data.get("ok"):
                self.logger.error(
                    f"Slack API Error: {response_data.get('error')}",
                    file=sys.__stderr__,
                )

        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}", file=sys.__stderr__)

    def _send_slack_with_retry(self, payload: Dict[str, Any], max_retries: int = 3, initial_delay: float = 1.0) -> Optional[Dict]:
        """
        Send a Slack message with retry logic for rate limiting.

        Args:
            payload (Dict[str, Any]): The message payload to send
            max_retries (int): Maximum number of retry attempts
            initial_delay (float): Initial delay in seconds between retries

        Returns:
            Optional[Dict]: Response data from Slack API if successful, None if all retries fail
        """
        delay = initial_delay
        attempt = 0

        while attempt < max_retries:
            try:
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={
                        "Authorization": f"Bearer {const.SLACK_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                
                response_data = response.json()
                
                if response_data.get("ok"):
                    return response_data
                
                if response_data.get("error") == "ratelimited":
                    # Get retry_after from headers or use exponential backoff
                    retry_after = float(response.headers.get("Retry-After", delay))
                    time.sleep(retry_after)
                    delay *= 2  # Exponential backoff
                    attempt += 1
                    continue
                
                # Other errors
                self.logger.error(
                    f"Slack API Error: {response_data.get('error')}",
                    file=sys.__stderr__,
                )
                return None

            except Exception as e:
                self.logger.error(f"Slack request failed: {e}", file=sys.__stderr__)
                attempt += 1
                time.sleep(delay)
                delay *= 2

        return None

    def set_level(self, level: str) -> None:
        """
        Update the logging level dynamically for all handlers.

        Args:
            level (str): New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
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
        """
        Set or update the pipeline name for Slack notifications.

        Args:
            name (str): Name of the pipeline
        """
        self._pipeline_name = name

    def set_output_file(self, log_file: str, overwrite: bool=True) -> None:
        """
        Change the log output file and reinitialize logger.

        Args:
            log_file (str): New log file path
        """
        self._log_file = log_file
        self.logger = self._setup_logger(overwrite=overwrite)

    def set_format(self, fmt: str) -> None:
        """
        Change the log message format and reinitialize logger.

        Args:
            fmt (str): New log message format
        """
        self._log_format = fmt
        self.logger = self._setup_logger()

class StdoutToLogger:
    """
    A file-like object that redirects stdout writes to a logger.

    This class enables capturing and logging of stdout output, 
    ensuring that error messages are properly tracked and 
    can be sent to multiple output streams.

    Attributes:
        logger (logging.Logger): Logger instance to redirect stdout
    """
    def __init__(self, logger):
        """
        Initialize StdoutToLogger with a logger.

        Args:
            logger (logging.Logger): Logger to use for stdout redirection
        """
        self.logger = logger

    def write(self, buf):
        """
        Write method to redirect stdout output to the logger.

        Args:
            buf (str): Buffer containing stdout output
        """
        for line in buf.rstrip().splitlines():
            self.logger.debug(line)
        
        # Also write to the actual stdout for console output
        sys.__stdout__.write(buf)

    def flush(self):
        """
        Flush method to maintain file-like object compatibility.
        """
        sys.__stdout__.flush()

class StderrToLogger:
    """
    A file-like object that redirects stderr writes to a logger.

    This class enables capturing and logging of stderr output, 
    ensuring that error messages are properly tracked and 
    can be sent to multiple output streams.

    Attributes:
        logger (logging.Logger): Logger instance to redirect stderr
    """
    def __init__(self, logger):
        """
        Initialize StderrToLogger with a logger.

        Args:
            logger (logging.Logger): Logger to use for stderr redirection
        """
        self.logger = logger

    def write(self, buf):
        """
        Write method to redirect stderr output to the logger.

        Args:
            buf (str): Buffer containing stderr output
        """
        for line in buf.rstrip().splitlines():
            self.logger.error(line)
        
        # Also write to the actual stderr for console output
        sys.__stderr__.write(buf)

    def flush(self):
        """
        Flush method to maintain file-like object compatibility.
        """
        sys.__stderr__.flush()

