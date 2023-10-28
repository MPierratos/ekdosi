import pathlib
import tempfile

__all__ = ["special_paths"]

module_name = __name__.split(".", maxsplit=1)[0]

PREFIX = "python--"


class ModuleSpecialPaths:

    """Common paths for the module to use

    temp_*: represent paths which sit on the operating system's (OS)
            temp directory. In Linux, an example would be `/tmp/`.
    user_*: represent paths which sit in the user's home folder.
            In Linux, an example would be `/home/<some_user>/Downloads`

            Note: user_share_* usually means $HOME/.local/share
            Note: user_config_* usually means $HOME/.config
            Note: user_cache_* usually means $HOME/.cache

    user_share -> Where user-specific data files should be stored.
    user_config -> Where user-specific configuration files should be stored.
    user_cache -> Where user-specific non-essental data files should be stored.
    temp_* is meant for general, non user specific storage with a short lifetime,
        where it is dependant upon the OS to clean. Usually on reboot, sometimes
        pruned on file lifetime.

    """

    _fields = [
        "DEFAULT_FOLDER_MODE",
        "tmp_folder",
        "tmp_folder_logs",
        "user_cache",
        "user_cache_logs",
        "user_share",
        "user_config",
        "user_downloads",
    ]

    _fields.sort()
    DEFAULT_FOLDER_MODE = 0o700

    tmp_folder = pathlib.Path(tempfile.gettempdir()) / f"{PREFIX}{module_name}"
    tmp_folder.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    tmp_folder_logs = tmp_folder / "logs"
    tmp_folder_logs.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_cache = pathlib.Path.home() / f".cache/{PREFIX}{module_name}"
    user_cache.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_cache_logs = user_cache / "logs"
    user_cache_logs.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_share = pathlib.Path.home() / f".local/share/{PREFIX}{module_name}"
    user_share.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_config = pathlib.Path.home() / f".config/{PREFIX}{module_name}"
    user_config.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_downloads = pathlib.Path.home() / f"Downloads/{PREFIX}{module_name}"
    user_downloads.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    user_secrets = pathlib.Path.home() / f".secrets/{PREFIX}{module_name}"
    user_secrets.mkdir(mode=DEFAULT_FOLDER_MODE, parents=True, exist_ok=True)

    __slots__ = []

    def __repr__(self) -> str:
        tmp = "Paths:\n\t"
        tmp += "\n\t".join(
            [f"{field:>20} -> {getattr(self, field)}" for field in self._fields]
        )
        return tmp


special_paths = ModuleSpecialPaths()

del pathlib
del tempfile
del module_name
