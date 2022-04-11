"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)

"""
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

def create_sample(file_name: str, line_numbers: int) -> None:
    """
    This function takes the path of the text file and the number of lines to be copied to the sample file
    """
    rows = []
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in range(line_numbers):
            rows.append(f.readline())
    with open(file_name + ".sample", 'w', newline="", encoding='utf-8', errors='ignore') as f:
        for line in rows:
            f.write(line)
