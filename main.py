from DigitalImaging import DigitalImaging
from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot

agriculture_img_path = "./assets/agriculture.jpg"
dog_folder_path = "./assets/dogs/"
d = DigitalImaging()

# Test reduce_to function
# img_red = d.reduce_to(agriculture_img_path, 'R')
# img_green = d.reduce_to(agriculture_img_path, 'G')
# img_blue = d.reduce_to(agriculture_img_path, 'B')
# img_red.show()
# img_green.show()
# img_blue.show()


# Test for make_collage function
Image.open(agriculture_img_path)
images = [
    Image.open(f'{dog_folder_path}bulldog.jpg'), Image.open(f'{dog_folder_path}cocker-spaniel.jpg'),
    Image.open(f'{dog_folder_path}continental-bulldog.jpg'), Image.open(f'{dog_folder_path}corgi.jpg'),
    Image.open(f'{dog_folder_path}dog.jpg'), Image.open(f'{dog_folder_path}dog.jpg'),
    Image.open(f'{dog_folder_path}dog3.jpg'),Image.open(f'{dog_folder_path}dog4.jpg'),
    Image.open(f'{dog_folder_path}maltese.jpg'), Image.open(f'{dog_folder_path}puppy.jpg'),
    Image.open(f'{dog_folder_path}puppy2.jpg'), Image.open(f'{dog_folder_path}puppy3.jpg'),

]

dogs_collage = d.make_collage(images)
# print(dogs_collage)

Image.fromarray(dogs_collage, 'RGB').show()
