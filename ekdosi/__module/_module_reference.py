import sys

__all__ = ["module_reference"]


class ModuleReference:
    name = __name__.split(".", maxsplit=1)[0]
    module = sys.modules[name]
    __slots__ = []


module_reference = ModuleReference()
