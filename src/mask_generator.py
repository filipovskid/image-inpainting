import numpy as np
import random


class MaskGenerator:
    @staticmethod
    def random_rectangle(img):
        mask = np.zeros(img.shape)
        h, w = img.shape[:2]
        img_area = w * h
        mask_area = 0
        offset = 10

        while 0.1 >= mask_area / img_area <= 0.4:
            x1, y1 = random.randrange(offset, int(h * 0.6)), random.randrange(offset, int(w * 0.6))
            x2, y2 = random.randrange(x1 + 1, h - offset), random.randrange(y1 + 1, w - offset)

            mask_area = (x2 - x1) * (y2 - y1)

        mask[x1:x2, y1:y2] = 1

        return mask

    @staticmethod
    def centered_rectangle(img):
        mask = np.zeros(img.shape)
        h, w = img.shape[:2]
        img_area = w * h
        mask_area = 0
        cx, cy = int(h / 2), int(w / 2)
        offset = 0
        min_pix = 20

        while not (0.1 <= (mask_area / img_area) <= 0.4):
            left = random.randrange(min_pix, int(w / 2))
            right = random.randrange(min_pix, int(w / 2))
            top = random.randrange(min_pix, int(h / 2))
            bottom = random.randrange(min_pix, int(h / 2))

            x1, y1 = cx - top, cy - left
            x2, y2 = cx + bottom, cy + right

            mask_area = (x2 - x1) * (y2 - y1)

        mask[x1:x2, y1:y2, :] = 1
        # mask[cx, cy] = 1

        return mask

    @staticmethod
    def random_noise(image, percent):
        row, col, ch = image.shape
        mask = np.zeros((row, col, 3))

        unif = np.random.sample((row, col))
        unif = unif.reshape(row, col)
        mask[unif > (1 - percent), :] = 1

        return mask