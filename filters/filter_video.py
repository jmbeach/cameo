import cv2

from filters.filter import Filter


class FilterVideo(Filter):
    def __init__(self, path):
        super(FilterVideo, self).__init__(duration=0)
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            self.logger.error(f"video file '{path}' doesn't exist")
            self.do_stop = True

    def draw(self, frame):
        if self.do_stop:
            return frame

        height, width, _ = frame.shape
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (width, height))
        return frame