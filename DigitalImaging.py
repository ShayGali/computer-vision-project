from dataclasses import dataclass
import dataclasses as dc
from typing import List

import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot


@dataclass
class DigitalImaging:
    def convert_to_gs(self, img_path: str):
        pass

    def color_at(self, img: np.ndarray, row_num: int, col_num: int):
        pass

    def reduce_to(self, path: str, RGB_char: str) -> Image:
        """
        get path for an image, and selected RGB color, and return new image only in the selected channel
        :param path: path to image
        :param RGB_char: can be - R, G, B
        :return: new Image object
        """
        valid_chars = ('R', 'G', 'B')
        if RGB_char not in valid_chars:
            raise ValueError(f"RGB_char need to be ('R', 'G', 'B'), you pass {RGB_char}")

        img_arr = np.array(Image.open(path))

        img_one_color_channel = img_arr.copy()

        if RGB_char == 'R':  # will take all the pixels and in channels 1 and 2 (green and blue), put the value 0
            img_one_color_channel[:, :, (1, 2)] = 0
        elif RGB_char == 'G':  # will take all the pixels and in channels 0 and 2 (red and blue), put the value 0
            img_one_color_channel[:, :, (0, 2)] = 0
        elif RGB_char == 'B':  # will take all the pixels and in channels 0 and 1 (red and green), put the value 0
            img_one_color_channel[:, :, (0, 1)] = 0

        return Image.fromarray(img_one_color_channel, 'RGB')  # convert the array to image

    def make_collage(self, images_list: List[Image.Image]) -> np.ndarray:
        """
        get list of images object and merge them to one collage.
        the collage will be -> 3 red channel images, 3 green channel images, 3 blue channel images and so on
        :param images_list: list on image object (from PIL.Image package)
        :return: np.array that represent the collage
        """
        # check that all images are type of Image.Image
        if any([not isinstance(img, Image.Image) for img in images_list]):
            raise ValueError("all images need to be of type PIL.Image")

        # convert the images from Image object to np.array
        images_list_as_array = [np.array(img) for img in images_list]  # convert all images to ndarray

        new_images = []  # all the new array will be store here

        for (index, img) in enumerate(images_list_as_array):
            if (index // 3) % 3 == 0:
                img[:, :, (1, 2)] = 0  # covert to red channel
            elif (index // 3) % 3 == 1:
                img[:, :, (0, 2)] = 0  # covert to green channel
            else:
                img[:, :, (0, 1)] = 0  # covert to blue channel
            new_images.append(img)

        return np.vstack(new_images)  # stack the np.arrays to one np.array vertically

    def shapes_dict(self):
        pass

    def detect_obj(self):
        pass

    def detect_obj_adv(self, detect_eyes, detect_faces):
        pass

    def detect_face_in_vid(self, video_path: str):
        pass
