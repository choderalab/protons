import os


def before_all(context):
    context.tmpfiles = []


def after_all(context):
    for filename in context.tmpfiles:
        os.remove(filename)
