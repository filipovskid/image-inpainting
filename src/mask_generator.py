import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch


class MaskGenerator:

    def __init__(self, seed=42):
        self.rs = RandomState(MT19937(SeedSequence(seed)))

    def random_rectangle(self, image_size, percent_range=(0.1, 0.5)):
        mask = np.ones(image_size)
        h, w = image_size
        img_area = w * h
        mask_area = 0
        offset = 10
        lower, upper = percent_range

        while not (lower <= mask_area / img_area <= upper):
            x1, y1 = self.rs.randint(offset, int(h * 0.6)), self.rs.randint(offset, int(w * 0.6))
            x2, y2 = self.rs.randint(x1 + 1, h - offset), self.rs.randint(y1 + 1, w - offset)

            mask_area = (x2 - x1) * (y2 - y1)

        mask[x1:x2, y1:y2] = 0

        return mask

    def centered_rectangle(self, image_size, percent_range=(0.1, 0.4)):
        mask = torch.ones(image_size)
        h, w = image_size
        img_area = w * h
        mask_area = 0
        cx, cy = int(h / 2), int(w / 2)
        offset = 0
        min_pix = round(min(h, w)/10)  # 20
        lower, upper = percent_range

        while not (lower <= (mask_area / img_area) <= upper):
            left = self.rs.randint(min_pix, int(w / 2))
            right = self.rs.randint(min_pix, int(w / 2))
            # right = left
            top = self.rs.randint(min_pix, int(h / 2))
            bottom = self.rs.randint(min_pix, int(h / 2))
            # bottom = top

            x1, y1 = cx - top, cy - left
            x2, y2 = cx + bottom, cy + right

            mask_area = (x2 - x1) * (y2 - y1)

        mask[x1:x2, y1:y2] = 0

        mask[int(h/2), int(w/2)] = 1

        return mask

    def rectangle_mask(self, image_size, props):
        img_size = image_size[1]
        mask = np.ones((img_size, img_size, 3))

        if props['mask_type'] == 'center':
            scale = 0.25
            low, upper = int(img_size * scale), int(img_size * (1.0 - scale))
            mask[low:upper, low:upper, :] = 0.
        elif props['mask_type'] == 'left':
            scale = 0.1
            left, right = int(img_size * scale), int(img_size * (1.0 - scale) * .5)
            top, bottom = int(img_size * .25), int(img_size * (1.0 - .25))

            mask[top:bottom, left:right, :] = 0.
        elif props['mask_type'] == 'right':
            scale = 0.1
            left, right = int(img_size * (1.0 + scale) * .5), int(img_size * (1.0 - scale))
            top, bottom = int(img_size * .25), int(img_size * (1.0 - .25))

            mask[top:bottom, left:right, :] = 0.

        return mask

    def random_noise(self, image_size, percent):
        row, col = image_size
        mask = np.ones((row, col, 3))

        unif = self.rs.random_sample((row, col))
        unif = unif.reshape(row, col)
        mask[unif > (1 - percent), :] = 0

        return mask