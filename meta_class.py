import logging


class MetaClass(type):
    """This meta class adds a logger property to class methods."""

    @property
    def logger(self):
        return logging.getLogger(self.__name__)