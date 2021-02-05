import cv2
import numpy

from filters.filter import Filter


class FilterColor(Filter):
    def __init__(self, color=(255, 0, 0)):
        super(FilterColor, self).__init__()
        self.color = color

    @staticmethod
    def add_color(frame, color):
        height, width, channels = frame.shape
        blank_image = numpy.zeros((height, width, channels), numpy.uint8)
        blank_image[:, :] = color
        frame = cv2.addWeighted(frame, 0.7, blank_image, 0.3, 0)
        return frame

    def draw(self, frame):
        frame = FilterColor.add_color(frame, self.color)
        return frame
