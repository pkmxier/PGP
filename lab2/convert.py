#!/usr/bin/env python3
from PIL import Image
import scipy.misc as smp
import numpy as np
import sys


mode = sys.argv[1]
input_image = sys.argv[2]
output_image = sys.argv[3]

if mode == "inverse":
    im = Image.open(input_image)
    with open(output_image, "wb") as output:
        w = im.size[0]
        h = im.size[1]
        output.write(w.to_bytes(4, byteorder='little'))
        output.write(h.to_bytes(4, byteorder='little'))

        for i in range(h):
            for j in range(w):
                rgb = im.getpixel((j, i))
                for color in rgb:
                    output.write(color.to_bytes(1, byteorder='little'))
                output.write((1).to_bytes(1, byteorder='little'))
else:
    with open(input_image, "rb") as input:
        w = int.from_bytes(input.read(4), byteorder='little')
        h = int.from_bytes(input.read(4), byteorder='little')
        data = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                rgba = [int.from_bytes(input.read(1), byteorder='little') for _ in range(4)]
                data[i, j] = rgba[:3]

        img = Image.fromarray(data)
        img.save(output_image)

