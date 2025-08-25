#import python's built-in logging library
import logging
#import os utilities for filesystem operations(like making folders)
import os
#import datetime to timespace log files with today's date
from datetime import datetime



#name of the folder where logs will be stored
LOG_DIR = "logs"
#create a folder named 'logs' if it doesn't already exist
os.makedirs(LOG_DIR, exist_ok=True)

#build a log file path like: logs/log_2023-10-05.log(changes daily)
LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

#configure the ROOt logger once for the whole program
logging.basicConfig(
    #write all logs to this file
    filename=LOG_FILE,
    #log message format
    # _%(asctime)s_: timestamp
    # _%(levelname)s: log level (INFO, ERROR, etc.)
    # _%(message)s: the actual log message text
    format="[%(asctime)s] %(levelname)s - %(message)s",
    #minimum level to record(info and above)
    level=logging.INFO,
)

def get_logger(name):
    """returns a named logger that inherits the root configuration above.
    use different names per module(e.g.,  __name__) to identify sources.
    """
    #get (or create) a logger with the given name 
    logger=logging.getLogger(name)
    #ensure this logger emits INFO and above(can be customized per logger)
    logger.setLevel(logging.INFO)
    #return the configured named logger
    return logger