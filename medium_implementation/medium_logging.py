# I got tired of my messy debugging so I cleaned it up

import sys

DEBUG = False

def set_debug(flag: bool) -> None:
    """Enable or disable debug printing."""
    global DEBUG
    DEBUG = bool(flag)


def dbg(msg: str) -> None:
    """Print debug message when DEBUG is True."""
    if DEBUG:
        print(str(msg))


def error(msg: str) -> None:
    """Always-print error messages to stderr."""
    print(str(msg), file=sys.stderr)
