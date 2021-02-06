import cv2
import numpy

from filters.filter_blur import FilterBlur


class FilterGlow(FilterBlur):
    def __init__(self, img_path):
        super(FilterBlur, self).__init__(duration=0)
        self.img_path = img_path
        self.do_stop = False
        self.debug = False

    def get_image(self, frame, faces):
        height, width, channels = frame.shape

        blur_strength = 30
        black = numpy.zeros((height, width, channels), numpy.uint8)
        black[:, :] = (0, 0, 0)
        white = numpy.zeros((height, width, channels), numpy.uint8)
        white[:, :] = (255, 255, 255)
        cutout = black.copy()
        feather_mask = numpy.zeros((height, width, channels), numpy.uint8)
        feather_mask[:, :] = (255, 255, 255)
        y_margin = 25
        x_margin = 30
        for x, y, w, h in faces:
            cutout[y - y_margin:height, x - x_margin:x + x_margin + w] += frame[y - y_margin:height,
                                                                          x - x_margin:x + x_margin + w]
            feather_mask[y - y_margin:height, x - x_margin:x + x_margin + w] += frame[y - y_margin:height,
                                                                                x - x_margin:x + x_margin + w]

        feather_mask = cv2.blur(feather_mask, (blur_strength, blur_strength))

        img = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height))
        feathered = FilterBlur.alpha_blend(frame, img, feather_mask)

        return feathered