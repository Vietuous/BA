import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, logfile: str, level=logging.INFO):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevents duplicate logging

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger
