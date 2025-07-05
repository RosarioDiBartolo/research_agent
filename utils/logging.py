# utils/logging.py

import logging
import sys
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal support
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)

        if "[PROMPT]" in record.msg:
            return Fore.LIGHTBLACK_EX + message + Style.RESET_ALL
        elif "[RESPONSE]" in record.msg:
            return Fore.CYAN + message + Style.RESET_ALL
        elif "[ERROR]" in record.msg:
            return Fore.RED + message + Style.RESET_ALL
        else:
            return message


def setup_logging():
    """Call this once at the top of your main script to configure colored + file logging."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter("%(asctime)s %(message)s"))

    file_handler = logging.FileHandler("model_invocations.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True  # Ensures reconfiguration if logging was already set up
    )
