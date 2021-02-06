import cv2
import numpy
import numpy as np

from filters.filter import Filter


class FilterBlur(Filter):
    """
    https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
    """

    face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')

    def __init__(self):
        super(FilterBlur, self).__init__(duration=0)
        self.do_stop = False
        self.debug = False

    @staticmethod
    def pixelize(frame, faces):
        """Pixelize each face."""

        height, width, channels = frame.shape

        blur_strength = 20
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

    @staticmethod
    # From https://stackoverflow.com/a/48274875/1834329
    def alpha_blend(a, b, mask):
        if mask.ndim == 3 and mask.shape[-1] == 3:
            alpha = mask / 255.0
        else:
            alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        return cv2.convertScaleAbs((a * (1 - alpha)) + (b * alpha))

    @staticmethod
    def areas_overlap(a, b):
        (x1, y1, w1, h1) = a
        (x2, y2, w2, h2) = b
        dx = min(x1+w1, x2+w2) - max(x1, x2)
        dy = min(y1+h1, y2+h2) - max(y1, y2)
        return dx >= 0 and dy >= 0

    @staticmethod
    def merge_faces(faces, width, frame):
        """
        If two faces overlap, merge them.

        This is way too complicated, but overlapping areas gets blurred twice
        otherwise. A much simpler workaround would be to keep the largest face.
        """

        if len(faces) == 1:
            return faces

        while True:
            overlap = False
            merged_faces = faces
            for i, a in enumerate(faces):
                (x1, y1, w1, h1) = a
                for j, b in enumerate(faces):
                    if j >= i:
                        continue
                    (x2, y2, w2, h2) = b
                    overlap = FilterBlur.areas_overlap(a, b)
                    if overlap:
                        # remove both faces
                        for x in [a, b]:
                            try:
                                merged_faces.remove(x)
                            except ValueError:
                                pass
                        # add merged faces
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = max(x1+w1, x2+w2) - x
                        h = max(y1+h1, y2+h2) - y
                        merged_faces.append((x, y, w, h))
                        break
                if overlap:
                    break

            faces = merged_faces
            if not overlap:
                break

        return faces

    @classmethod
    def detect_faces(cls, frame, cascade):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) == 0:
            cls.logger.debug("no face detected")
            height, width, _ = frame.shape
            faces = [ (0, 0, width, height) ]
        return faces

    @staticmethod
    def increase_face_area(faces, width, height, margin=80):
        return [ (max(0, x - margin // 2),
                  max(0, y - margin // 2),
                  min(w + margin, width),
                  min(h + margin, height))
                 for x, y, w, h in faces ]

    @staticmethod
    def mark_faces(frame, faces, rgb=(255, 255, 0)):
        """Add a rectangle around each face."""

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), rgb, 2)

    def draw(self, frame):
        faces = FilterBlur.detect_faces(frame, FilterBlur.face_cascade)
        height, width, _ = frame.shape
        faces = FilterBlur.increase_face_area(faces, width, height)
        faces = FilterBlur.merge_faces(faces, width, frame)
        frame = FilterBlur.pixelize(frame, faces)
        if self.debug:
            FilterBlur.mark_faces(frame, faces)
        self.last_frame = frame

        return frame