from src.pipeline.runner import run_pipeline
from src.loggers import setup_loggers
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    setup_loggers("logs")
    logger = logging.getLogger("main")

    try:
        run_pipeline("config")
        logger.info("OMR pipeline finished successfully.")

    except Exception as e:
        logger.critical(f"Error in main pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
