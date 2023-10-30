import logging

import ekdosi.__module as __module
import ekdosi.settings as settings

__all__ = ["show"]


def show() -> None:
    """banner"""
    MSG = f"Status: {__module.module_reference.name} loaded!"
    settings.logging.console_logger.set_verbose()
    logger = logging.getLogger(__name__)
    logger.info(
        "\n".join(
            [
                str(xxx)
                for xxx in [
                    MSG,
                ]
            ]
        )
    )
    settings.logging.console_logger.set_simple()
