from DigitalImaging import DigitalImaging

from PIL import Image
import numpy as np
# import matplotlib.pyplot as pyplot
# import cv2


d = DigitalImaging()

# Test for convert_to_gs function
# img_converted_gs = d.convert_to_gs('assets/dogs/puppy3.jpg')
# img_converted_gs.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Test for color_at function
# img_as_array = np.array(Image.open('assets/dogs/puppy3.jpg'))
# # img_as_array.flags.writeable = False # if we change the flag the method will return None
# rgb_colors = d.color_at(img_as_array, img_as_array.shape[0]/2, img_as_array.shape[1]/2)
# print(rgb_colors)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Test reduce_to function
# agriculture_img_path = "./assets/agriculture.jpg"
# img_red = d.reduce_to(agriculture_img_path, 'R')
# img_green = d.reduce_to(agriculture_img_path, 'G')
# img_blue = d.reduce_to(agriculture_img_path, 'B')
# img_red.show()
# img_green.show()
# img_blue.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Test for make_collage function
# images = DigitalImaging.list_of_all_img_in_folder("assets/dogs")
# dogs_collage = d.make_collage(images)
# Image.fromarray(dogs_collage, 'RGB').show()


# Test for shapes_dict function
# images = DigitalImaging.list_of_all_img_in_folder("assets/different image height")
# for i in d.shapes_dict(images).items():
#     print(i)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Test for detect_obj function
# img_arr_eyes = d.detect_obj('assets/peoples/people3.jpg', "eyes")
# img_arr_faces = d.detect_obj('assets/peoples/people3.jpg', "face")
# DigitalImaging.show_cv2_img(img_arr_eyes, "eyes detect")
# DigitalImaging.show_cv2_img(img_arr_faces, "face detect")

# Test for detect_obj_adv function
# img_arr = d.detect_obj_adv('assets/peoples/people3.jpg', True, False)
# DigitalImaging.show_cv2_img(img_arr, "findings")

# Test for detect_face_in_vid function
# d.detect_face_in_vid("path")