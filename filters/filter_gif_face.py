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
        self.background = cv2.imread('./gif_background.png', cv2.IMREAD_UNCHANGED)
        self.frame = 0
        self.total_frames = 0
        self.imgs = []
        while self.total_frames < 1:
            try:
                self.gif.seek(self.gif.tell() + 1)
                self.gif.save(f'./gif_frame_{self.frame}.png')
                self.imgs.append(cv2.imread(f'./gif_frame_{self.frame}.png', cv2.IMREAD_UNCHANGED))
                self.frame += 1
            except:
                self.gif.close()
                self.total_frames = self.frame
                self.frame = 0

    def get_image(self, frame, faces):
        height, width, channels = frame.shape
        img = self.imgs[self.frame]
        img_height, img_width, img_channels = img.shape
        black_a = numpy.zeros((height, width, 4), numpy.uint8)
        black_a[:, :] = (0, 0, 0, 0)
        mask_a = black_a.copy()
        for x, y, w, h in faces:
            x_offset = -20
            y_offset = -40
            start_y = y + y_offset
            end_y = min(start_y + img_height, height)
            draw_height = end_y - start_y
            start_x = x + x_offset
            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0
            end_x = min(start_x + img_width, width)
            draw_width = end_x - start_x
            mask_a[start_y:end_y, start_x:end_x] += img[0 if end_y >= draw_height else draw_height - end_y:draw_height, 0:draw_width]
            alphas = mask_a.copy()
            alphas = np.where(alphas < (10, 10, 10, 10), black_a, alphas)
            mask = cv2.cvtColor(alphas, cv2.COLOR_BGRA2BGR)
            frame = np.where(mask == (0, 0, 0), frame, mask)
        self.frame += 1
        if self.frame >= self.total_frames:
            self.frame = 0
        return frame
