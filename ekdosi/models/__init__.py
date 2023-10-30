from typing import Dict
import sys
from ekdosi.errors import ModelNotFoundError

__all__ = ["BERTEmbedding", "BERT", "register_model"]


# specifies a dictionary of architectures
_MODELS: Dict[str, any] = {}  # registry


def register_model(name: str):
    """Decorator used register a model architecture
    Args:
        name: Name of the model
    """

    def register_class(cls, name: str):
        _MODELS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


def models_in_registry() -> list:
    """return list of model names in the registry"""
    return list(_MODELS.keys())


def get_from_registry(model_name: str):
    """get a model from the registry

    Args:
        model_name (str): model name, i.e. lowercase version of the class name
    """
    if model_name in _MODELS.keys():
        return _MODELS.get(model_name)
    else:
        raise ModelNotFoundError("Model not found in the registry.")


from ekdosi.models._bert import BERTEmbedding, BERT, test_bert

# cleanup
del sys
del Dict
