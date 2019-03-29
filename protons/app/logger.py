# coding=utf-8
import logging


# Import log and use it in your code to log messages to the same logger throughout.
log = logging.getLogger()
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)5s() ] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.INFO)
