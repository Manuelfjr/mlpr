import sys
from pathlib import Path


def set_root(level: int = 1) -> Path:
    """Set the root directory of the project.

    Parameters:
    -----------
    level : int, optional
        The number of parent directories to go up from the current file's directory.
        Default is 1, which means the immediate parent directory.

    Returns:
    --------
    Path
        The path to the root directory of the project.

    Examples:
    ---------
    >>> set_root()
    PosixPath('/user/dir/path/root/project-path')

    >>> set_root(2)
    PosixPath('/user/dir/path/root')

    """
    for i in range(level):
        if i == 0:
            PROJECT_DIR = Path(__file__).parent
        else:
            PROJECT_DIR = PROJECT_DIR.parent
    sys.path.append(str(PROJECT_DIR))
    return PROJECT_DIR
