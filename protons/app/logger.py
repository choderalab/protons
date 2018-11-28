# coding=utf-8
import logging

log = logging.getLogger()
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)5s() ] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.INFO)

