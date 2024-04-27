import yaml


def read_file_yaml(path: str) -> dict:
    """Read a YAML file and return its contents as a dictionary.

    Parameters:
    -----------
    path : str
        The path to the YAML file.

    Returns:
    --------
    dict
        A dictionary containing the contents of the YAML file.

    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist.

    yaml.YAMLError
        If there is an error parsing the YAML file.

    Examples:
    ---------
    >>> read_file_yaml('/path/to/file.yaml')
    {'key1': 'value1', 'key2': 'value2'}
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_file_yaml(data: dict, path: str) -> None:
    """Save a dictionary as a YAML file.

    Parameters:
    -----------
    path : str
        The path to save the YAML file.
    data : dict
        The dictionary to be saved as YAML.

    Raises:
    -------
    PermissionError
        If there is a permission error while saving the file.

    Examples:
    ---------
    >>> data = {'key1': 'value1', 'key2': 'value2'}
    >>> save_file_yaml('/path/to/file.yaml', data)
    """
    with open(path, "w") as file:
        yaml.dump(data, file)
    return None
