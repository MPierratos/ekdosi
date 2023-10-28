"""Storage spot for terminal codes and some colour schemes"""

__all__ = ["TerminalVisualCodes", "NullColourFormatBase", "DefaultColourFormat"]


class TerminalVisualCodes:
    """
    This class contains attributes for terminal visual codes.

    Attributes:
    st_reset: Reset all styles to default
    st_bold: Bold text
    st_faint: Faint text
    st_italic: Italic text
    st_underline: Underline text
    st_slow_blink: Slow blink text
    st_rapid_blink: Rapid blink text
    st_swap_fg_bg: Swap foreground and background colours
    st_hide_text: Hide text
    st_strikethrough: Strikethrough text
    fg_black: Foreground colour black
    fg_red: Foreground colour red
    fg_green: Foreground colour green
    fg_yellow: Foreground colour yellow
    fg_blue: Foreground colour blue
    fg_magenta: Foreground colour magenta
    fg_cyan: Foreground colour cyan
    fg_white: Foreground colour white
    bg_black: Background colour black
    bg_red: Background colour red
    bg_green: Background colour green
    bg_yellow: Background colour yellow
    bg_blue: Background colour blue
    bg_magenta: Background colour magenta
    bg_cyan: Background colour cyan
    bg_white: Background colour white
    fg_bright_black: Bright foreground colour black
    fg_bright_red: Bright foreground colour red
    fg_bright_green: Bright foreground colour green
    fg_bright_yellow: Bright foreground colour yellow
    fg_bright_blue: Bright foreground colour blue
    fg_bright_magenta: Bright foreground colour magenta
    fg_bright_cyan: Bright foreground colour cyan
    fg_bright_white: Bright foreground colour white
    bg_bright_black: Bright background colour black
    bg_bright_red: Bright background colour red
    bg_bright_green: Bright background colour green
    bg_bright_yellow: Bright background colour yellow
    bg_bright_blue: Bright background colour blue
    bg_bright_magenta: Bright background colour magenta
    bg_bright_cyan: Bright background colour cyan
    bg_bright_white: Bright background colour white
    """

    st_reset = 0
    st_bold = 1
    st_faint = 2
    st_italic = 3
    st_underline = 4
    st_slow_blink = 5
    st_rapid_blink = 6
    st_swap_fg_bg = 7
    st_hide_text = 8
    st_strikethrough = 9
    fg_black = 30
    fg_red = 31
    fg_green = 32
    fg_yellow = 33
    fg_blue = 34
    fg_magenta = 35
    fg_cyan = 36
    fg_white = 37
    bg_black = 40
    bg_red = 41
    bg_green = 42
    bg_yellow = 43
    bg_blue = 44
    bg_magenta = 45
    bg_cyan = 46
    bg_white = 47
    fg_bright_black = 90
    fg_bright_red = 91
    fg_bright_green = 92
    fg_bright_yellow = 93
    fg_bright_blue = 94
    fg_bright_magenta = 95
    fg_bright_cyan = 96
    fg_bright_white = 97
    bg_bright_black = 100
    bg_bright_red = 101
    bg_bright_green = 102
    bg_bright_yellow = 103
    bg_bright_blue = 104
    bg_bright_magenta = 105
    bg_bright_cyan = 106
    bg_bright_white = 107

    @staticmethod
    def print_fg_colours():
        """
        This method prints all the foreground colour codes with their respective colours.
        """
        for i in range(30, 38):
            print(f"\033[{i}m This is colour code {i}")
        for i in range(90, 98):
            print(f"\033[{i}m This is colour code {i}")


tvs = TerminalVisualCodes


class NullColourFormatBase:
    """Class that has an attribute with the same name as each logging levels
    in lowercase python identifiers. Also includes additional attributes for
    accent, date, message, diminish.
    """

    reset = f"\033[{tvs.st_reset}m"
    debug = ""
    info = ""
    warning = ""
    error = ""
    critical = ""
    accent = ""
    date = ""
    message = ""
    diminish = ""


class DefaultColourFormat(NullColourFormatBase):
    debug = f"\033[{tvs.st_reset};{tvs.fg_bright_yellow};{tvs.bg_black}m"
    info = f"\033[{tvs.st_reset};{tvs.fg_green}m"
    warning = f"\033[{tvs.st_reset};{tvs.fg_bright_magenta}m"
    error = f"\033[{tvs.st_reset};{tvs.fg_black};{tvs.bg_red}m"
    critical = f"\033[{tvs.st_rapid_blink};{tvs.fg_black};{tvs.bg_red}m"
    accent = f"\033[{tvs.st_bold};{tvs.fg_bright_green};{tvs.bg_black}m"
    message = f"\033[{tvs.st_reset};{tvs.fg_bright_black}m"
    date = f"\033[{tvs.st_reset};{tvs.fg_bright_black}m"
    diminish = f"\033[{tvs.st_reset};{tvs.fg_bright_black}m"


del tvs
