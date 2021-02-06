import cv2
import numpy
import numpy as np

from filters.filter import Filter
from filters.filter_blur import FilterBlur


class FilterBackgroundBlur(FilterBlur):
    def get_image(self, frame, faces):
        height, width, channels = frame.shape

        blur_strength = 25
        black = numpy.zeros((height, width, channels), numpy.uint8)
        black[:, :] = (0, 0, 0)
        white = numpy.zeros((height, width, channels), numpy.uint8)
        white[:, :] = (255, 255, 255)
        cutout = black.copy()
        feather_mask = numpy.zeros((height, width, channels), numpy.uint8)
        feather_mask[:, :] = (255, 255, 255)
        y_margin = 25
        for x, y, w, h in faces:
            cutout[y - y_margin:height, x:x + w] += frame[y - y_margin:height, x:x + w]
            feather_mask[y - y_margin:height, x:x + w] -= white[y - y_margin:height, x:x + w]

        feather_mask = cv2.blur(feather_mask, (blur_strength, blur_strength))

        blurred = frame.copy()
        blurred = cv2.blur(blurred, (blur_strength, blur_strength))
        feathered = FilterBlur.alpha_blend(frame, blurred, feather_mask)

        return feathered
