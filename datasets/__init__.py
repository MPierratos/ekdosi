import os
from ekdosi.settings import special_paths
import pathlib

__all__ = ["SquadDataSet"]

def _default_data_dir() -> pathlib.Path:
    """sets the data dir to the user cache as the default dir
    
    i.e. ~/.cache/python--ekdosi
    """
    return special_paths.user_cache
   
def _get_data_dir() -> pathlib.Path | None:
    """get the data directory from the env variable DATA_DIR

    Returns:
        pathlib.Path: data dir path
    """

    dir = os.getenv("DATA_DIR", None)
    if dir:
        return pathlib.Path(dir)
    else:
        return None

DATA_DIR = _default_data_dir() if _get_data_dir() is None else _get_data_dir()

from ._squad_dataset import SquadDataset, test_squad_dataset

del os
del special_paths
del pathlib

