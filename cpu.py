import numpy as np
import sys
import scipy.misc as smp


def intensity(rgb):
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114


input_image = sys.argv[1]
output_image = sys.argv[2]

with open(input_image, "rb") as input, open(output_image, "wb") as output:
    w = int.from_bytes(input.read(4), byteorder='big')
    h = int.from_bytes(input.read(4), byteorder='big')
    data = np.zeros((h, w), dtype=np.int16)
    dst = np.zeros((h, w, 3), dtype=np.int16)

    for i in range(h):
        for j in range(w):
            r = int.from_bytes(input.read(4), byteorder='big')
            g = int.from_bytes(input.read(4), byteorder='big')
            b = int.from_bytes(input.read(4), byteorder='big')
            alpha = int.from_bytes(input.read(4), byteorder='big')
            data[i, j] = intensity([r, g, b])
    for i in range(h - 1):
        for j in range(w - 1):
            Gx = data[i, j] - data[i + 1, j + 1]
            Gy = data[i, j + 1] - data[i + 1, j]
            dst[i, j] = [min((Gx ** 2 + Gy ** 2) ** 0.5, 255) for _ in range(3)]
    img = smp.toimage(dst)
    img.show()
