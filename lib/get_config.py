import configparser
import os

def get_config(config_path='config.ini'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)

    return config