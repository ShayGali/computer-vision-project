import glob
from dataclasses import dataclass
import dataclasses as dc
from typing import List, Literal, Tuple
import numpy as np
from PIL import Image
import cv2


@dataclass
class DigitalImaging:
    def convert_to_gs(self, img_path: str) -> Image.Image:
        """
        Converts image to grey scale image using PIL,
        prints new image mode & returns grey scale Image Object

        :param img_path: path to image location
        :return: img_as_gs: image object of given img_path converted to grey scale
        """
        img = Image.open(img_path)
        img_as_gs = img.convert('L')
        print(f"Image mode before convert - '{img.mode}'")
        return img_as_gs

    def color_at(self, img: np.ndarray, row_num: int, col_num: int) -> Tuple[int, int, int] | None:
        """
        This method gets an image as numpy array,
        also gets a row number and a column number, which represent a specific pixel in the image.
        Checks if the image is writeable -
        if true, the method returns a tuple of the RGB value of the pixel that
        been given, by the row number and column number.
        else, the method returns None.

        :param img: image as numpy array
        :param row_num: pixel row number
        :param col_num: pixel column number
        :return: tuple with the color of the pixel in rgb format
        """
        if img.flags.writeable:
            a_img = Image.fromarray(img, 'RGB')
            RGB = a_img.getpixel((row_num, col_num))
            a_tuple = tuple(RGB)
            return (a_tuple)
        return None

    def reduce_to(self, path: str, RGB_char: str) -> Image.Image:
        """
        Gets a path for an image & selected RGB color, and return new image only in the selected channel

        :param path: path to image location
        :param RGB_char: can be - R, G, B
        :return: new Image object
        """
        valid_chars = ('R', 'G', 'B')
        if RGB_char not in valid_chars:
            raise ValueError(
                f"RGB_char need to be ('R', 'G', 'B'), you pass {RGB_char}")

        img_arr = np.array(Image.open(path))

        img_one_color_channel = img_arr.copy()

        # will take all the pixels and in channels 1 and 2 (green and blue), put the value 0
        if RGB_char == 'R':
            img_one_color_channel[:, :, (1, 2)] = 0
        # will take all the pixels and in channels 0 and 2 (red and blue), put the value 0
        elif RGB_char == 'G':
            img_one_color_channel[:, :, (0, 2)] = 0
        # will take all the pixels and in channels 0 and 1 (red and green), put the value 0
        elif RGB_char == 'B':
            img_one_color_channel[:, :, (0, 1)] = 0

        # convert the array to image
        return Image.fromarray(img_one_color_channel, 'RGB')

    def make_collage(self, images_list: List[Image.Image]) -> np.ndarray:
        """
        Gets list of images object and merge them to one collage.
        the collage will be -> 3 red channel images, 3 green channel images, 3 blue channel images and so on

        :param images_list: list on image object (from PIL.Image package)
        :return: np.array that represent the collage
        """
        # check that all images are type of Image.Image
        if any([not isinstance(img, Image.Image) for img in images_list]):
            raise ValueError("all images need to be of type PIL.Image")

        # resize the images to be the same size
        # the resize will be to the max width and the max height of the images
        max_width_img = max(images_list, key=lambda img: img.width).width
        max_height_img = max(images_list, key=lambda img: img.height).height
        images_list = [image.resize((max_width_img, max_height_img)) for image in images_list]

        # convert the images from Image object to np.array
        # convert all images to ndarray
        images_list_as_array = [np.array(img) for img in images_list]

        new_images = []  # all the new array will be store here

        for (index, img) in enumerate(images_list_as_array):
            if (index // 3) % 3 == 0:
                img[:, :, (1, 2)] = 0  # covert to red channel
            elif (index // 3) % 3 == 1:
                img[:, :, (0, 2)] = 0  # covert to green channel
            else:
                img[:, :, (0, 1)] = 0  # covert to blue channel
            new_images.append(img)

        # stack the np.arrays to one np.array vertically
        return np.vstack(new_images)

    def shapes_dict(self, img_list: List[Image.Image]) -> dict:
        # each item will look like -> filename : shape(height,width,num_of_channels)
        shape_dict = {}  # dict of filename to shape tuple

        for img in img_list:  # loop over the image list
            img_as_arr = np.asarray(img)
            img_shape = img_as_arr.shape
            img_filename = img.filename
            shape_dict[img_filename] = img_shape

        # return the dict sorted by the img height
        # item[1] - the image dimensions (shape)
        # item[1][0] - the height of the img
        return {key: value for key, value in sorted(shape_dict.items(), key=lambda item: item[1][0])}

    def detect_obj(self, img_path: str, detect_location: Literal["eyes", "face"]) -> np.ndarray:
        """
        Searches a given image for eyes/faces & returns an Image object
        with all findings surrounded by a green rectangle border.

        :param img_path: path to the image location
        :param detect_location: Which part to detect. can be "face" or "eyes"
        :return: Image object with green rectangle surrounding the findings
        """
        # the valid detect locations
        valid_detect_locations = ("face", "eyes",)

        # check the type of the img_path
        if not isinstance(img_path, str):
            raise TypeError(
                f"img_path need to be of type str, you passed {type(img_path)}")

        # convert the detect_location to lower case
        detect_location = detect_location.lower()

        # check if the detect_location is in the valid_detect_locations
        if detect_location not in valid_detect_locations:
            raise ValueError(
                f"detect_location can be ({valid_detect_locations}), you passed {detect_location}")

        # select the classifiers according to the detect_location
        if detect_location == "face":
            classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif detect_location == "eyes":
            classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml")

        img_as_arr = cv2.imread(img_path)  # read the img
        img_in_gray = \
            cv2.cvtColor(
                img_as_arr, cv2.COLOR_BGR2GRAY)  # make a copy of the image in gray scale (for better performance)

        img_faces = classifier.detectMultiScale(
            img_in_gray)  # detect the object

        # paint the rectangle around the objects that detected
        for (row, column, width, height) in img_faces:
            cv2.rectangle(img_as_arr,  # image
                          (row, column),  # upper left corner of each face
                          # lower right corner of each face
                          (row + width, column + height),
                          (0, 255, 0),  # paint the rectangle in green (BGR)
                          2)

        if len(img_faces) == 0:  # if we don't find anything
            print(f"{detect_location} not detected")

        return cv2.cvtColor(img_as_arr, cv2.COLOR_BGR2RGB)

    def detect_obj_adv(self, img_path: str, detect_eyes: bool, detect_faces: bool) -> np.ndarray:
        """
        Detects objects of faces and eyes & surrounds them with a green border triangle, depends on params which object
        would be searched for.
        Similar to "detect_obj" function.

        :param img_path: path to image location
        :param detect_eyes: boolean for searching eyes
        :param detect_faces: boolean for searching faces
        :return: Image object with green rectangle surrounding the findings
        """
        # check the type of the img_path
        if not isinstance(img_path, str):
            raise TypeError(f"img_path need to be of type str, you passed {type(img_path)}")

        classifiers = []

        if detect_eyes:
            classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml"))
        if detect_faces:
            classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
        if not detect_faces and not detect_eyes:
            raise TypeError("Detect faces & eyes are both False. For the method to work one has to be True")

        # read the img
        img_as_arr = cv2.imread(img_path)

        # make a copy of the image in gray scale (for better performance)
        img_in_gray = cv2.cvtColor(img_as_arr, cv2.COLOR_BGR2GRAY)

        # Runs over the classifiers and paints green border rectangles over detected objects
        for classifier in classifiers:
            detect_objects = classifier.detectMultiScale(img_in_gray)
            # paint the rectangle around the objects that detected
            for (row, column, width, height) in detect_objects:
                cv2.rectangle(img_as_arr,  # image
                              (row, column),  # upper left corner of each face
                              # lower right corner of each face
                              (row + width, column + height),
                              (0, 255, 0),  # paint the rectangle in green (BGR)
                              2)

            if len(detect_objects) == 0:  # If no objects were detected
                print("No face/eyes objects detected")

        return cv2.cvtColor(img_as_arr, cv2.COLOR_BGR2RGB)

    def detect_face_in_vid(self, video_path: str) -> None:
        """
        This method gets a path of a video clip, as a string and detects all the faces in that video.
        :param video_path: a path of a video clip as a string
        :return: None
        """
        # check the type of the video_path
        if not isinstance(video_path, str):
            raise TypeError(f"video_path need to be of type str, you passed {type(video_path)}")
        vid = cv2.VideoCapture(video_path)
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            frame_in_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(frame_in_gray)
            for (row, col, w, h) in faces:
                cv2.rectangle(frame, (row, col), (row + w, col + h), (0, 255, 0), 2)
            frame = cv2.resize(frame, (650, 350))
            cv2.imshow('video', frame)
            if cv2.waitKey(1) == 27:
                break

        vid.release()
        cv2.destroyAllWindows()

    @staticmethod
    def list_of_all_img_in_folder(file_path: str) -> List[Image.Image]:
        """
        get a path to folder and return a list on PIL.Image Object.
        all the file need to be a image file
        :param file_path: path to foldr
        :return: list of PIL.Image
        """
        return [Image.open(f) for f in glob.iglob(file_path + "/*.jpg")]
