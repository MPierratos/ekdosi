__all__ = ["MissingEnvironmentVariableError",
           "ModelNotFoundError"]


class MissingEnvironmentVariableError(Exception):
    """Raised when searching for an environment variable and it is not found."""

    def __init__(self, message: str):
        super().__init__(message)

class ModelNotFoundError(Exception):
    """Raise an exception when trying to load a model from the registry"""

    def __init__(self, message: str):
        super().__init__(message)