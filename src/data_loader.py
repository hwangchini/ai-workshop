"""Utility for loading and parsing data from external JSON files."""

import json
import logging
from typing import Union, Dict, Any # Import Union for older Python versions

logger = logging.getLogger(__name__)

def _load_json_file(file_path: str) -> Union[Dict[str, Any], None]:
    """A generic function to load a JSON file and handle common errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Loading data from {file_path}")
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}", exc_info=True)
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
        return None

def load_medical_personnel_data(path: str = "data/medical_personnel.json") -> Union[Dict[str, Any], None]:
    """Loads the medical personnel data from the specified JSON file."""
    return _load_json_file(path)

def load_patient_data(path: str = "data/patient_data.json") -> Union[Dict[str, Any], None]:
    """Loads the patient data from the specified JSON file."""
    return _load_json_file(path)

def load_symptoms_data(path: str = "data/symptoms.json") -> Union[Dict[str, Any], None]:
    """Loads the symptoms data from the specified JSON file."""
    return _load_json_file(path)
