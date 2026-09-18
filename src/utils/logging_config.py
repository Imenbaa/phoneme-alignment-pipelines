import logging

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(log_file, mode="w"),logging.StreamHandler()])

