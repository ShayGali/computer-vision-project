from DigitalImaging import DigitalImaging
from PIL import Image
import numpy as np

agriculture_img_path = "./assets/agriculture.jpg"
d = DigitalImaging()

# Test reduce_to function
img_red = d.reduce_to(agriculture_img_path, 'R')
img_green = d.reduce_to(agriculture_img_path, 'G')
img_blue = d.reduce_to(agriculture_img_path, 'B')
img_red.show()
img_green.show()
img_blue.show()
