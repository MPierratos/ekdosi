"""Custom logger implementation"""

import logging
import logging.handlers
import re
from logging import Formatter

# Local Imports
import ekdosi.__module as __module
from ekdosi.settings.logging._terminal_colours import (
    DefaultColourFormat,
    NullColourFormatBase,
)

DISK_RECORD_START = "#"
DISK_RECORD_FIELD_DELIMETER = "-"
DISK_RECORD_END = "#"

console_simple_fmt = (
    "%(asctime)s [%(levelname)8s][PID:%(process)s][%(name)s.%(lineno)03s] %(message)s"
)
console_verbose_fmt = "[%(levelname)8s][%(asctime)s][PID:%(process)s][%(processName)s.%(threadName)s]\n[%(name)s.%(lineno)04s]\n%(message)s\n"

__all__ = ["console_logger"]


class CustomFormatterColourized(Formatter):
    def __init__(
        self,
        format,
        colour_format: NullColourFormatBase = NullColourFormatBase,
        datefmt: str = "%H:%M:%S%z",
        strip_colour_codes: bool = False,
    ):
        self.__format = format
        self.__colour_format = colour_format
        self.strip_colour_codes = strip_colour_codes
        super().__init__(format, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Fomat the specific record as text.

        Args:
            record (logging.LogRecord): The record to be formatted.

        Returns:
            str: the formatted record
        """

        # Dynamically set the colour of the levelname based on the level
        levelname = record.levelname.lower()
        record.start_char = DISK_RECORD_START
        record.delimiter = DISK_RECORD_FIELD_DELIMETER
        record.end_char = DISK_RECORD_START

        if hasattr(self.__colour_format, levelname):
            colour = getattr(self.__colour_format, levelname)
            N = max(8 - len(record.levelname), 0)
            lname = f"{self.__colour_format.reset}{colour}{record.levelname}{self.__colour_format.reset}"
            record.levelname = " " * N + lname

        if hasattr(record, "lineno"):
            record.lineno = f"{self.__colour_format.accent}{record.lineno}{self.__colour_format.reset}"

        if hasattr(record, "message"):
            record.message = f"{self.__colour_format.message}{record.message}{self.__colour_format.reset}"

        if hasattr(record, "asctime"):
            record.asctime = f"{self.__colour_format.date}{record.asctime}{self.__colour_format.reset}"
        if hasattr(record, "threadName"):
            record.threadName = f"{self.__colour_format.accent}{record.threadName}{self.__colour_format.reset}"
        if hasattr(record, "process"):
            record.process = f"{self.__colour_format.accent}{record.process}{self.__colour_format.reset}"

        loglet = super().format(record) + self.__colour_format.reset

        if self.strip_colour_codes:
            loglet = re.sub(r"\x1b\[\d+(;\d+)*m", "", loglet)
        return loglet


console_simple_fmt_colourless = CustomFormatterColourized(
    format=console_simple_fmt, colour_format=NullColourFormatBase, datefmt="%H:%M"
)
console_verbose_fmt_colourless = CustomFormatterColourized(
    format=console_verbose_fmt,
    colour_format=NullColourFormatBase,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
console_simple_fmt_default = CustomFormatterColourized(
    format=console_simple_fmt, colour_format=DefaultColourFormat, datefmt="%H:%M"
)
console_verbose_fmt_default = CustomFormatterColourized(
    format=console_verbose_fmt,
    colour_format=DefaultColourFormat,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

# Create handlers for the custom formatters
console_simple_handler = logging.StreamHandler()
console_simple_handler.setFormatter(console_simple_fmt_default)

console_verbose_handler = logging.StreamHandler()
console_verbose_handler.setFormatter(console_verbose_fmt_default)

# Get the logger for __module.ref.name
logger = logging.getLogger(__module.module_reference.name)

# set the default level of the logger to debug
logger.setLevel(logging.DEBUG)

# add the handlers to the logger
logger.addHandler(console_simple_handler)


class ConsoleLoggerSettings:
    __slots__ = []

    def set_simple(self, colour: bool = True) -> None:
        """Toggle the logger to use the console simple handler"""
        logger.removeHandler(console_verbose_handler)
        if colour:
            console_simple_handler.setFormatter(console_simple_fmt_default)
        else:
            console_simple_handler.setFormatter(console_simple_fmt_colourless)

        logger.addHandler(console_simple_handler)

    def set_verbose(self, colour: bool = True) -> None:
        """Toggle the logger to use the console verbose handler"""
        logger.removeHandler(console_simple_handler)
        if colour:
            console_verbose_handler.setFormatter(console_verbose_fmt_default)
        else:
            console_verbose_handler.setFormatter(console_verbose_fmt_colourless)

        logger.addHandler(console_verbose_handler)

    def remove_all_custom_formatting_handlers(self) -> None:
        """Removes all custmo formatting handlers from the logger."""
        logger.removeHandler(console_simple_handler)
        logger.removeHandler(console_verbose_handler)


console_logger = ConsoleLoggerSettings()
