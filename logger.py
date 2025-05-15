import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    handlers=[
        logging.FileHandler("logfile.log"), 
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger()
logger.info("This is testing log")
