import os


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
