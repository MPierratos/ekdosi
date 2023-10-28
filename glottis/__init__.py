import glottis.__module as __module

__all__ = ["settings", "errors", "configs","models"]


def load_local_modules() -> None:
    """This function loads the local submodules

    In particular, it supports timing the individual load times of
    the submodules.
    """
    SUBMODULE_IMPORT_ORDER = ["settings", 
                              "errors", 
                              "configs",  
                              "models"]

    __module.utils.ImportTimerBulk(
        module_names=SUBMODULE_IMPORT_ORDER,
        packages=__name__,
        print_to_stdout=True,  # elsewise logs to the debugger
    ).import_modules()


load_local_modules()

# tidy up
del load_local_modules
