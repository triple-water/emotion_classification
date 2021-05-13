import logging
import os


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'func_name_override'):
            record.funcName = record.func_name_override
        if hasattr(record, 'file_name_override'):
            record.filename = record.file_name_override
        return super(CustomFormatter, self).format(record)


def get_logger(log_file_name, log_sub_dir=""):
    log_dir = '../logs/'
    log_dir = os.path.join(log_dir, log_sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logPath = log_file_name if os.path.exists(log_file_name) else os.path.join(log_dir, (str(log_file_name) + '.log'))
    logger = logging.Logger(log_file_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(logPath, 'a+')
    handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s - %(message)s'))
    logger.addHandler(handler)
    return logger
