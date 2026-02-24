import os
import logging
import logging.config
import logging.handlers
import json
import time
from pathlib import Path
from typing import Optional, Any, Dict, Union, List
import sys
from datetime import datetime
import atexit


class ConditionalFileHandler(logging.FileHandler):
    """
    A file handler that only creates the file if messages are actually logged to it.
    """
    
    def __init__(self, filename, mode='a', encoding=None, delay=True, min_level=None, max_level=None):
        """
        Initialize the handler with delayed file creation.
        
        Args:
            filename: Name of the log file
            mode: File mode (default 'a' for append)
            encoding: File encoding
            delay: Whether to delay file creation until first log message
            min_level: Minimum level to log (inclusive)
            max_level: Maximum level to log (inclusive), None means no limit
        """
        # Store the original filename for later use
        self._original_filename = filename
        self._file_created = False
        self._min_level = min_level
        self._max_level = max_level
        super().__init__(filename, mode, encoding, delay=True)
        
    def emit(self, record):
        """
        Emit a record, creating the file only when needed.
        Only emit if the record level matches our criteria.
        
        Args:
            record: LogRecord to emit
        """
        # Check level filtering
        if self._min_level is not None and record.levelno < self._min_level:
            return
        if self._max_level is not None and record.levelno > self._max_level:
            return
            
        if not self._file_created:
            self._file_created = True
            # File will be created by parent class on first emit
        super().emit(record)


class LoggerManager:
    """
    Enhanced logger manager for gfmbench_api with better configuration handling.
    """
    
    def __init__(self, config_file: str = 'log.json'):
        """
        Initialize the logger manager.
        
        Args:
            config_file: Name of the logging configuration file
        """
        self.config_file = config_file
        self.config_path = Path(__file__).parent / config_file
        self.log_dir = None
        self.is_initialized = False
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.conditional_handlers = {}  # Track conditional handlers
        self.created_files = []  # Track created log files
        
    def ensure_log_directory(self, log_dir: str = 'logs') -> Path:
        """
        Ensure log directory exists and return its path.
        
        Args:
            log_dir: Name of the log directory
            
        Returns:
            Path object of the log directory
        """
        cwd = Path.cwd()
        log_path = cwd / log_dir
        log_path.mkdir(exist_ok=True)
        self.log_dir = log_path
        return log_path
        
    def get_timestamped_filename(self, base_filename: str) -> str:
        """
        Generate a timestamped filename using session timestamp.
        
        Args:
            base_filename: Base filename (with or without extension)
            
        Returns:
            Timestamped filename
        """
        # Remove existing .log extension if present
        if base_filename.endswith('.log'):
            base_filename = base_filename[:-4]
            
        return f"{base_filename}_{self.session_timestamp}.log"
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load logging configuration from JSON file.
        
        Returns:
            Dictionary containing logging configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file has invalid JSON
        """
        try:
            with open(self.config_path, 'r') as json_file:
                config = json.load(json_file)
                self._validate_config(config)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Logging config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file {self.config_path}: {e}")
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the logging configuration for common issues.
        
        Args:
            config: Logging configuration dictionary
        """
        # Check for required top-level keys
        required_keys = ['version', 'handlers', 'formatters']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in logging configuration")
                
        # Check handlers for common issues
        handlers = config.get('handlers', {})
        for handler_name, handler_config in handlers.items():
            handler_class = handler_config.get('class', '')
            
            # Validate file handlers have required parameters
            if 'FileHandler' in handler_class:
                if 'filename' not in handler_config:
                    raise ValueError(f"Handler '{handler_name}' missing 'filename' parameter")
                    
            # Check for valid formatter references
            formatter = handler_config.get('formatter')
            if formatter and formatter not in config.get('formatters', {}):
                raise ValueError(f"Handler '{handler_name}' references unknown formatter '{formatter}'")
            
    def update_handler_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update file paths for all file-based handlers in the configuration.
        
        Args:
            config: Logging configuration dictionary
            
        Returns:
            Updated configuration dictionary
        """
        log_path = self.ensure_log_directory()
        handlers = config.get('handlers', {})
        
        for handler_name, handler_config in handlers.items():
            handler_class = handler_config.get('class', '')
            
            # Update file-based handlers
            if 'FileHandler' in handler_class and 'filename' in handler_config:
                base_filename = handler_config['filename']
                # Always use timestamped filenames for uniqueness
                filename = self.get_timestamped_filename(base_filename)
                handler_config['filename'] = str(log_path / filename)
                
        return config
        
    def create_conditional_handler(self, handler_type: str, level: str, formatter_name: str = 'detailed') -> Optional[logging.Handler]:
        """
        Create a conditional file handler that only creates files when needed.
        
        Args:
            handler_type: Type of handler ('error' or 'debug')
            level: Logging level for the handler
            formatter_name: Name of the formatter to use
            
        Returns:
            ConditionalFileHandler instance
        """
        log_path = self.ensure_log_directory()
        base_filename = f"gfmbench_{handler_type}"
        filename = self.get_timestamped_filename(base_filename)
        full_path = str(log_path / filename)
        
        # Set level filtering based on handler type
        min_level = None
        max_level = None
        
        if handler_type == 'error':
            min_level = logging.ERROR  # Only ERROR and CRITICAL
        elif handler_type == 'debug':
            min_level = logging.DEBUG
            max_level = logging.DEBUG  # Only DEBUG messages
        
        # Create conditional handler with level filtering
        handler = ConditionalFileHandler(full_path, mode='w', min_level=min_level, max_level=max_level)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Set formatter
        formatters = {
            'detailed': logging.Formatter(
                '[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(filename)s:%(lineno)d] [%(funcName)s()] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'debug': logging.Formatter(
                '[%(asctime)s] [%(levelname)-8s] [%(process)d:%(thread)d] [%(name)s] [%(filename)s:%(lineno)d] [%(funcName)s()] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        }
        
        if formatter_name in formatters:
            handler.setFormatter(formatters[formatter_name])
        
        # Store reference
        self.conditional_handlers[handler_type] = handler
        
        return handler
        
    def init_logger(self, log_level: Optional[str] = None) -> logging.Logger:
        """
        Initialize the logging system with enhanced configuration.
        
        Args:
            log_level: Optional log level override
            
        Returns:
            Configured root logger
        """
        try:
            # Load and update configuration
            config = self.load_config()
            config = self.update_handler_paths(config)
            
            # Override log level if specified
            if log_level:
                config['root']['level'] = log_level.upper()
                
            # Apply configuration
            logging.config.dictConfig(config)
            
            # Create conditional handlers
            error_handler = self.create_conditional_handler('error', 'ERROR', 'detailed')
            debug_handler = self.create_conditional_handler('debug', 'DEBUG', 'debug')
            
            # Add conditional handlers to root logger and main loggers
            root_logger = logging.getLogger('gfmbench')
            root_logger.addHandler(error_handler)
            root_logger.addHandler(debug_handler)
            
            # Set up third-party library loggers (these are now in config, but ensure they're set)
            self._configure_third_party_loggers()
            
            self.is_initialized = True
            root_logger.info("Logging system initialized successfully")
            root_logger.info(f"Log directory: {self.log_dir}")
            root_logger.info(f"Session timestamp: {self.session_timestamp}")
            
            # Register cleanup function
            atexit.register(self._cleanup_empty_log_files)
            
            return root_logger
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] [%(levelname)-8s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            logging.error(f"Failed to initialize enhanced logging: {e}")
            logging.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__cause__') and e.__cause__:
                logging.error(f"Caused by: {e.__cause__}")
            logging.info("Using basic logging configuration as fallback")
            logging.info(f"Log directory attempted: {self.log_dir}")
            return logging.getLogger()
            
    def _configure_third_party_loggers(self):
        """Configure third-party library loggers to reduce noise."""
        third_party_loggers = {
            'PIL': logging.WARNING,
            'matplotlib': logging.WARNING,
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'tensorflow': logging.ERROR,
            'torch': logging.WARNING,
        }
        
        for logger_name, level in third_party_loggers.items():
            try:
                logging.getLogger(logger_name).setLevel(level)
            except Exception:
                pass  # Ignore if logger doesn't exist
                
    def _cleanup_empty_log_files(self):
        """
        Clean up empty log files at program exit.
        """
        for handler_type, handler in self.conditional_handlers.items():
            if hasattr(handler, '_file_created') and not handler._file_created:
                # Handler was created but never used, file doesn't exist yet
                continue
                
            if hasattr(handler, 'baseFilename'):
                log_file = Path(handler.baseFilename)
                if log_file.exists() and log_file.stat().st_size == 0:
                    try:
                        log_file.unlink()  # Remove empty file
                        print(f"Removed empty log file: {log_file.name}")
                    except OSError:
                        pass  # Ignore errors during cleanup
                        
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (if None, returns 'gfmbench' logger)
            
        Returns:
            Logger instance
        """
        if not self.is_initialized:
            self.init_logger()
            
        if name is None:
            name = 'gfmbench'
        elif not name.startswith('gfmbench'):
            name = f'gfmbench.{name}'
            
        return logging.getLogger(name)


# Global logger manager instance
_logger_manager = LoggerManager()


def init_logger(log_level: Optional[str] = None) -> logging.Logger:
    """
    Initialize the logging system (convenience function).
    
    Args:
        log_level: Optional log level override
        
    Returns:
        Configured logger
    """
    return _logger_manager.init_logger(log_level)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance (convenience function).
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return _logger_manager.get_logger(name)


def get_log_prefix(log_path: Optional[str] = None) -> str:
    """
    Gets the prefix of a log file name by removing the extension.
    
    Args:
        log_path: Path to the log file. If None, uses the current logger's file
        
    Returns:
        Log file name prefix without extension
    """
    try:
        if log_path is None:
            # Get the first file handler from the current logger
            logger = logging.getLogger()
            for handler in logger.handlers:
                if hasattr(handler, 'baseFilename'):
                    log_path = handler.baseFilename
                    break
        else:
            return "unknown"
                
        log_filename = Path(log_path).name
        return log_filename.rsplit('.', 1)[0]  # Remove extension
        
    except Exception:
        return "unknown"


def get_stamp_from_log() -> str:
    """
    Get the time stamp from the log file.
    Matches the original implementation: split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")
    
    Returns:
        Filename without extension from the first file handler, or empty string if not found
    """
    try:
        # Get the first file handler from the current logger (matching original behavior)
        logger = logging.getLogger()
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                # This matches: split(handler.baseFilename)[-1].replace(".log","")
                return Path(handler.baseFilename).name.replace(".log", "")
        
        return ""
        
    except Exception:
        return ""


def record_input_message(name: str, value: Any, logger: Optional[logging.Logger] = None) -> None:
    """
    Records a debug log message for an input parameter and its value.
    
    Args:
        name: Name of the input parameter
        value: Value of the input parameter
        logger: Optional logger instance (if None, uses default logger)
    """
    if logger is None:
        logger = get_logger()
        
    if value is not None:
        # Better formatting for different types
        if isinstance(value, (dict, list)):
            logger.debug(f"Parameter {name}: {type(value).__name__} with {len(value)} items")
        elif isinstance(value, str) and len(value) > 100:
            logger.debug(f"Parameter {name}: {type(value).__name__} (length: {len(value)})")
        else:
            logger.debug(f"Parameter {name}: {value}")


def get_target_dir(name: str, base_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Gets or creates a target directory.
    
    Args:
        name: Name of the directory to create
        base_path: Base path for the directory (if None, uses current working directory)
        
    Returns:
        Path object of the target directory
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
        
    target_dir = base_path / name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None, 
                     logger: Optional[logging.Logger] = None) -> None:
    """
    Log function call details for debugging.
    
    Args:
        func_name: Name of the function being called
        args: Positional arguments
        kwargs: Keyword arguments
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger()
        
    if kwargs is None:
        kwargs = {}
        
    args_str = ", ".join(str(arg)[:50] for arg in args)  # Limit length
    kwargs_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in kwargs.items())
    
    call_str = f"{func_name}("
    if args_str:
        call_str += args_str
    if kwargs_str:
        call_str += (", " if args_str else "") + kwargs_str
    call_str += ")"
    
    logger.debug(f"Function call: {call_str}")


def log_execution_time(func_name: str, start_time: float, 
                      logger: Optional[logging.Logger] = None) -> None:
    """
    Log execution time of a function.
    
    Args:
        func_name: Name of the function
        start_time: Start time (from time.time())
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger()
        
    execution_time = time.time() - start_time
    logger.info(f"Function '{func_name}' completed in {execution_time:.4f} seconds")


def setup_exception_logging() -> None:
    """
    Set up global exception handling to log uncaught exceptions.
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger = get_logger()
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception