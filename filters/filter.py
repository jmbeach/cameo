import datetime
import logging
from abc import abstractmethod
from meta_class import MetaClass


class Filter(metaclass=MetaClass):
    def __init__(self, duration=1.0):
        self.start_time = datetime.datetime.now()
        self.duration = duration
        self.do_stop = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("start")

    def stop(self):
        self.do_stop = True

    def done(self):
        diff = datetime.datetime.now() - self.start_time
        done = (self.duration > 0 and diff.total_seconds() >= self.duration) or self.do_stop
        if done:
            self.logger.info("stop")
        return done

    @abstractmethod
    def draw(self, frame):
        pass