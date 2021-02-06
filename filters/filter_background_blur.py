import cv2
import numpy
import numpy as np

from filters.filter import Filter


class FilterBackgroundBlur(Filter):
    def __init__(self):
        super(FilterBackgroundBlur, self).__init__(duration=0)
        self.do_stop = False

    # From https://www.learnpythonwithrune.org/opencv-python-a-simple-approach-to-blur-the-background-from-webcam/
    def draw(self, frame):
        # convert bgr to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask based on medium to high saturation and value
        # the lower values can be changed to match your environment.
        h = int(cv2.getTrackbarPos('h', 'Preview'))
        s = int(cv2.getTrackbarPos('s', 'Preview'))
        v = int(cv2.getTrackbarPos('v', 'Preview'))
        height, width, channels = frame.shape
        blank_image = numpy.zeros((height, width, channels), numpy.uint8)
        blank_image[:, :] = (255, 0, 211)
        mask = cv2.inRange(hsv, (h, s, v), (180, 255, 255))
        # _, mask = cv2.threshold(hsv, 100, 255, cv2.THRESH_BINARY)

        # copy the mask 3 times to fit the frames
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Make a frame that's blurred
        # blurred_frame = cv2.cvtColor(frame, (25, 25))

        # Combine the original with the blurred frame using the mask
        frame = np.where(mask_3d == (255, 255, 255), frame, blank_image)
        return frame
