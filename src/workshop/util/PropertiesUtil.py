import logging

def load_properties(file_path: str) -> dict:
    """Loads a .properties file into a dictionary."""
    props = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    props[key.strip()] = value.strip()
    except FileNotFoundError:
        # Use logging.warning instead of print for consistency
        logging.warning(f"Properties file not found at {file_path}. Using default values.")
    except Exception as e:
        # Catch other potential errors during file reading
        logging.error(f"Error loading properties file {file_path}: {e}", exc_info=True)
    return props
