#we want to define our general modules here
from configparser import ConfigParser
from argparse import ArgumentParser

def get_configs(file_path: str) -> ConfigParser:
    """
    this function takes the path of the config file, reads it, and return the content
    """
    parser = ConfigParser()
    parser.read(file_path)
    return parser

def get_args() -> ArgumentParser:
    """
    this function reads the runtime arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, dest="config_path", default="")
    return parser
