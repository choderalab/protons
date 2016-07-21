import logging

log = logging.getLogger()
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
log.setLevel(logging.INFO)

