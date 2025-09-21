import numpy as np
import torch

import random
import logging
import logging.config
import os
import yaml

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="./conf/logging.yaml", default_level=logging.INFO, log_dir=None
):
    """Set up configuration for logging utilities.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to YAML file containing configuration for Python logger,
        by default "./conf/logging.yaml"
    default_level : logging object, optional, by default logging.INFO
        log_dir : str, optional
        Directory to store log files, by default None (uses directory in config)
    """

    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        # Modify log file paths if log_dir is provided
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            for handler in log_config["handlers"].values():
                if "filename" in handler:
                    # Extract just the filename from the path
                    filename = os.path.basename(handler["filename"])
                    # Create new path with the provided directory
                    handler["filename"] = os.path.join(log_dir, filename)

        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.error(error)
        logger.error("Logging config file is not found. Basic config is being used.")
