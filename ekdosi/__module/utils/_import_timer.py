import importlib
import logging
import typing as ty
from time import perf_counter

logger = logging.getLogger(__name__)

__all__ = ["Timer", "ImportTimer", "ImportTimerBulk"]


class Timer:
    def __init__(self, title: str, print_to_stdout: bool = False) -> None:
        self.title = title
        self.print_to_stdout = print_to_stdout

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = perf_counter()
        msg = f"{self.title}: {end_time - self.start_time:.3f} (sec)"
        logger.debug(msg)
        match self.print_to_stdout:
            case True:
                print(msg)
            case _:
                pass
        return self


class ImportTimer:
    def __init__(
        self,
        module_name: str,
        package: ty.Optional[str] = None,
        print_to_stdout: bool = False,
    ) -> None:
        """Initialize the ImportTimer class.

        Args:
            module_name (str): The name of the module to import
            package (ty.Optional[str], optional): The package for which to import the module. Defaults to None.
            print_to_stdout (bool, optional): Whether to print the import time to stdout. Defaults to False.
        """
        self.module_name = module_name
        self.package = package
        self.print_to_stdout = print_to_stdout

    def import_module(self) -> ty.Any:
        """Import the module and measure the time it takes

        Returns:
            ty.Any: The imported module
        """

        start_time = perf_counter()

        thing = importlib.import_module(self.module_name, package=self.package)

        end_time = perf_counter()

        module_name = (
            f"{self.package}{self.module_name}" if self.package else self.module_name
        )

        msg = f"Submodule Loadtime: {end_time - start_time:.3f} (sec) {module_name}"

        if self.print_to_stdout:
            print(msg)
        else:
            logger.debug(msg)

        return thing


class ImportTimerBulk(ImportTimer):
    def __init__(
        self,
        module_names: list,
        packages: ty.Union[str, list] = None,
        print_to_stdout: bool = False,
    ) -> None:
        assert isinstance(module_names, list)
        match packages:
            case None:
                packages = [None] * len(module_names)
            case [_]:
                assert len(module_names) == len(packages)
            case str():
                packages = [packages] * len(module_names)
            case _:
                raise NotImplementedError(
                    "Unknown mixture of types for module_names and packages"
                )
        assert len(module_names) > 0
        assert len(packages) > 0
        self.module_names = module_names
        self.packages = packages
        self.print_to_stdout = print_to_stdout

    def import_modules(self, strict: bool = True):
        modules = []
        for self.module_name, self.package in zip(
            self.module_names, self.packages, strict=strict
        ):
            if self.package is not None and self.module_name[0] != ".":
                # relative import
                self.module_name = f".{self.module_name}"
            importer = ImportTimer(
                module_name=self.module_name,
                package=self.package,
                print_to_stdout=self.print_to_stdout,
            )
            modules.append(importer.import_module())
        return modules
