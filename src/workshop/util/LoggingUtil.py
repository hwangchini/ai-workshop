"""Utility for configuring the application's logging system and providing activity logging decorators."""

import logging
import os
import datetime
from logging.handlers import RotatingFileHandler
from functools import wraps

def _archive_previous_logs():
    """Checks for existing log files from a previous run and moves them to a history folder."""
    log_dir = "logs"
    history_dir = os.path.join(log_dir, "history")
    log_files = ["system.log", "app.log", "conversation.log"]

    if not os.path.exists(log_dir):
        return

    primary_log = os.path.join(log_dir, "system.log")
    if os.path.exists(primary_log):
        mod_time = os.path.getmtime(primary_log)
        timestamp = datetime.datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
        
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        for log_file in log_files:
            old_path = os.path.join(log_dir, log_file)
            if os.path.exists(old_path):
                new_filename = f"{timestamp}-{log_file}"
                new_path = os.path.join(history_dir, new_filename)
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    print(f"Could not archive log file {old_path}: {e}")

# --- Part 1: Logging Setup ---

def setup_logging():
    """Archives old logs and sets up new, clean logging handlers for the current session."""
    _archive_previous_logs()

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # --- 1. Root Logger (for system errors and general console output) ---
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    system_log_format = "[%(asctime)s][%(name)s][%(funcName)s:%(lineno)d] %(levelname)s - %(message)s"
    system_formatter = logging.Formatter(system_log_format)
    
    system_file_handler = RotatingFileHandler(os.path.join(log_dir, "system.log"), maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    system_file_handler.setLevel(logging.ERROR)
    system_file_handler.setFormatter(system_formatter)
    root_logger.addHandler(system_file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # --- 2. Conversation Logger (for chat history) ---
    conversation_logger = logging.getLogger('conversation')
    if conversation_logger.hasHandlers():
        conversation_logger.handlers.clear()
    conversation_logger.setLevel(logging.INFO)
    conversation_logger.propagate = False
    
    conversation_log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    conversation_formatter = logging.Formatter(conversation_log_format)
    
    conversation_file_handler = RotatingFileHandler(os.path.join(log_dir, "conversation.log"), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    conversation_file_handler.setLevel(logging.INFO)
    conversation_file_handler.setFormatter(conversation_formatter)
    conversation_logger.addHandler(conversation_file_handler)

    # --- 3. App Activity Logger (for method entry/exit) ---
    activity_logger = logging.getLogger('app_activity')
    if activity_logger.hasHandlers():
        activity_logger.handlers.clear()
    activity_logger.setLevel(logging.INFO)
    activity_logger.propagate = False
    
    activity_log_format = "[%(asctime)s] [%(name)s] - %(message)s"
    activity_formatter = logging.Formatter(activity_log_format)
    
    activity_file_handler = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    activity_file_handler.setLevel(logging.INFO)
    activity_file_handler.setFormatter(activity_formatter)
    activity_logger.addHandler(activity_file_handler)

    # --- 4. Redirect Third-Party Library Logs to app.log ---
    # Get the loggers for the libraries that are producing the HTTP logs and redirect them.
    third_party_loggers_to_redirect = ["openai", "httpx"]
    for logger_name in third_party_loggers_to_redirect:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(activity_file_handler) # Redirect their output to app.log
        logger.propagate = False # IMPORTANT: Stop them from printing to the console via the root logger

    logging.getLogger(__name__).info("Logging configured. Previous logs archived.")


# --- Part 2: Activity Logging Decorator ---

def log_activity(func):
    """A decorator to log the entry and exit of a method to the app_activity log file."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        local_activity_logger = logging.getLogger('app_activity')

        try:
            class_name = args[0].__class__.__name__
        except (IndexError, AttributeError):
            class_name = ""
        
        method_name = func.__name__
        full_name = f"{class_name}.{method_name}" if class_name else method_name

        local_activity_logger.info(f"Entering method: {full_name}")
        try:
            result = func(*args, **kwargs)
            local_activity_logger.info(f"Exiting method: {full_name} - Success")
            return result
        except Exception as e:
            local_activity_logger.error(f"Exiting method: {full_name} - Failed with error: {e}", exc_info=True)
            raise # Re-raise the exception after logging
    return wrapper
