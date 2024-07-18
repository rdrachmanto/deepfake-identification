import os
import json


def create_directory(parent: str, children: list[str] | None) -> None:
    """
    Create a directory and its children (if filled)
    Args:
    - `parent (str)`: The parent directory path
    - `children (list[str] | None)`: Optional list of subdirectories
    """

    if children == None:
        os.makedirs(f"{parent}")
        return

    for c in children:
        if not os.path.exists(f"{parent}/{c}"):
            os.makedirs(f"{parent}/{c}")


def preprocess_fail_stat(path: str, preprocessor: str, dataset: str):
    """
    Generate json object from log files about frame extraction failure, 
    containing file information, failed frame number and counts 
    Args:
    - `path (str)`: Path to the log file to examine
    - `preprocessor (str)`: Filter the preprocessor (MTCNN or Dlib)
    - `dataset (str)`: Filter the dataset 
    """
    with open(path) as f:
        lines = f.read().splitlines()
        for l in lines:
            print(l)

