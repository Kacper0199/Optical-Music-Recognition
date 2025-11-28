import logging
import sys
import os
from datetime import datetime


def setup_loggers(logs_root):
    if not os.path.exists(logs_root):
        os.makedirs(logs_root)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_file_path = os.path.join(logs_root, f"run_{timestamp}.txt")

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] :: %(asctime)s :: %(name)s :: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)
