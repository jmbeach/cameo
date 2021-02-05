import cv2
import numpy

from filters.filter import Filter
from filters.filter_color import FilterColor


class FilterAddImage(Filter):
    def __init__(self, path):
        super(FilterAddImage, self).__init__(duration=2)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.logger.error(f"image file '{path}' doesn't exist")
            self.do_stop = True
            return

        self.img = cv2.resize(img, (150, 150))
        self.height2, self.width2, _ = self.img.shape

    def draw(self, frame):
        if self.do_stop:
            return frame

        frame = FilterColor.add_color(frame, (255, 178, 0))

        # add an alpha channel
        b_channel, g_channel, r_channel = cv2.split(frame)
        alpha_channel = numpy.ones(b_channel.shape, dtype=b_channel.dtype) * 50
        frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        height, width, channels = frame.shape

        # resize to the frame size and center the image
        blank = numpy.zeros((height, width, channels), numpy.uint8)
        x, y = (width - self.width2) // 2, (height - self.height2) // 2
        blank[y:y+self.height2, x:x+self.width2] += self.img

        # create a mask from the given image
        img2gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # apply this mask to the frame: this region is now black
        frame_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)

        # add the image
        frame = cv2.add(frame_bg, blank)

        return frame