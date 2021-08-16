import logging
import sys
from typing import List, Union

import dill


class LoggerFormatter(logging.Formatter):
    """TODO: use relative path for %(pathname)s"""

    default_time_format = "%m/%d/%Y %H:%M:%S"
    default_msec_format = "%s,%02d"
    FORMATS = {
        logging.DEBUG: (
            "%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s"
        ),
        logging.INFO: "%(asctime)s - [%(name)s.%(levelname)s] -- %(message)s",
        logging.WARNING: "%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s",
        logging.ERROR: (
            "%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s"
        ),
        "DEFAULT": "%(asctime)s - %(levelname)s -- %(message)s",
    }

    def __init__(self, fmt="%(asctime)s - %(levelname)s -- %(message)s"):
        LoggerFormatter.FORMATS["DEFAULT"] = fmt
        super().__init__(fmt=fmt, datefmt=None, style="%")

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        _fmt = getattr(self._style, "_fmt")
        # Replace the original format with one customized by logging level
        setattr(self._style, "_fmt", self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"]))
        # Call the original formatter class to do the grunt work
        fmt = logging.Formatter.format(self, record)
        # Restore the original format configured by the user
        setattr(self._style, "_fmt", _fmt)
        return fmt


def get_logger(
    logger_id: str, file: Union[str, List[str]] = None, console: bool = False
) -> logging.Logger:
    # NOTE: logging.getLogger create new instance based on `name`
    # no new instance will be created if the same name is provided
    logger = logging.getLogger(str(logger_id))
    logger.setLevel(logging.DEBUG)

    fmt = LoggerFormatter()
    # create console handler and set level to the vebosity
    SH = list(filter(lambda h: isinstance(h, logging.StreamHandler), logger.handlers))
    if len(SH) == 0 and console:  # if console handler is not registered yet
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # create file handler and set level to debug
    FH = list(filter(lambda h: isinstance(h, logging.FileHandler), logger.handlers))
    if file is not None and len(FH) == 0:
        file = [file] if isinstance(file, str) else file
        for f in set(file) - set(fh.baseFilename for fh in FH):
            try:
                fh = logging.FileHandler(f)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except FileNotFoundError as _:
                pass

    logger.propagate = False
    return logger


def dump_logger(logger: logging.Logger) -> str:
    FHs = list(filter(lambda h: isinstance(h, logging.FileHandler), logger.handlers))
    log_file = None if len(FHs) == 0 else [fh.baseFilename for fh in FHs]
    SH = list(filter(lambda h: isinstance(h, logging.StreamHandler), logger.handlers))
    return dill.dumps({"logger_id": logger.name, "console": len(SH) != 0, "file": log_file})


def load_logger(logger_str: str) -> logging.Logger:
    return get_logger(**dill.loads(logger_str))
