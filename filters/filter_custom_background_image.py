import cv2
import numpy
import numpy as np
from PIL import Image

from filters.filter_blur import FilterBlur


class FilterGifFace(FilterBlur):
    def __init__(self, img_path):
        super(FilterBlur, self).__init__(duration=0)
        self.img_path = img_path
        self.do_stop = False
        self.debug = False
        self.gif = Image.open(self.img_path)
        self.gif.save('./gif_background.png')

    def get_image(self, frame, faces):
        height, width, channels = frame.shape
        try:
            self.gif.seek(self.gif.tell() + 1)
        except:
            self.gif.close()
            self.gif = Image.open(self.img_path)

        self.gif.save('./gif_frame.png')
        img = cv2.imread('./gif_frame.png', cv2.IMREAD_UNCHANGED)
        background = cv2.imread('./gif_background.png', cv2.IMREAD_UNCHANGED)
        img_height, img_width, img_channels = img.shape
        if img_height > height or img_width > width:
            img_height = int(img_height * 0.5)
            img_width = int(img_width * 0.5)
            img = cv2.resize(img, (img_height, img_width))
            background = cv2.resize(background, (img_height, img_width))
        for x, y, w, h in faces:
            for x_img in range(img_width):
                for y_img in range(img_height):
                    if y + y_img >= height or x + x_img >= width:
                        continue
                    bg_pixel = background[y_img, x_img]
                    if len(bg_pixel > 3) and bg_pixel[3] < 255:
                        # don't draw bg
                        'do nothing'
                    else:
                        frame[y + y_img, x + x_img] = (bg_pixel[0], bg_pixel[1], bg_pixel[2])
                    pixel = img[y_img, x_img]
                    if len(pixel) > 3 and pixel[3] < 255:
                        continue
                    frame[y + y_img, x + x_img] = (pixel[0], pixel[1], pixel[2])
        return frame

class FilterCustomBackgroundImage(FilterBlur):
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
        x_margin = 40
        for x, y, w, h in faces:
            cutout[y - y_margin:height, x - x_margin:x + x_margin + w] += frame[y - y_margin:height,
                                                                          x - x_margin:x + x_margin + w]
            feather_mask[y - y_margin:height, x - x_margin:x + x_margin + w] -= white[y - y_margin:height,
                                                                                x - x_margin:x + x_margin + w]

        feather_mask = cv2.blur(feather_mask, (blur_strength, blur_strength))

        img = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height))
        feathered = FilterBlur.alpha_blend(frame, img, feather_mask)

        filtered_cutout = cutout.copy()
        filtered_cutout = cv2.cvtColor(filtered_cutout, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame, (155, 135, 0), (255, 255, 255))
        # copy the mask 3 times to fit the frames
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # Combine the original with the blurred frame using the mask
        filtered_cutout = np.where(mask_3d == (255, 255, 255), img, feathered)
        touchup_fraction_face_y_top = 8
        touchup_fraction_face_x = 4
        touchup_fraction_torso_x = 10
        for x, y, w, h in faces:
            # ensure inner part of face doesn't get masked
            filtered_cutout[y + int(h / touchup_fraction_face_y_top): (y + h) - int(h / touchup_fraction_face_y_top), x + int(w / touchup_fraction_face_x): (x + w) - int(w / touchup_fraction_face_x)] = frame[y + int(h / touchup_fraction_face_y_top): (y + h) - int(h / touchup_fraction_face_y_top), x + int(w / touchup_fraction_face_x): (x + w) - int(w / touchup_fraction_face_x)]

            # ensure inner part of torso not cropped
            filtered_cutout[(y + h) - int(h / touchup_fraction_face_y_top):height, x + int(w / touchup_fraction_torso_x): (x + w) - int(w / touchup_fraction_torso_x)] = frame[(y + h) - int(h / touchup_fraction_face_y_top):height, x + int(w / touchup_fraction_torso_x): (x + w) - int(w / touchup_fraction_torso_x)]

        return np.where(filtered_cutout == (0, 0, 0), img, filtered_cutout)