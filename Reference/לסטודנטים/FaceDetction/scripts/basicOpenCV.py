import cv2
import matplotlib.pyplot as plt

# pip3 install matplotlib
def show_image(name):
    while True:
        cv2.imshow('image', name)
        key_pressed = cv2.waitKey(0)
        # if key_pressed & 27: # by default
        if key_pressed & ord('q'): # q character is pressed
            break;
    # cv2.destroyWindow('image') # release image window resources
    cv2.destroyAllWindows()


mayim_as_img = cv2.imread('../resources/mayim.jpg')
# show_image(mayim_as_img)
cast_image = cv2.imread('../resources/cast.jpg')
# show_image(cast_image)
mayim_rgb = cv2.cvtColor(mayim_as_img,cv2.COLOR_BGR2RGB)

plt.imshow(mayim_rgb) # show image using pyplot
plt.show()

"""
Face detection vs. Face Recognition
Face detection - detect human faces according to a set of features common to all human faces
Face Recognition - detect if there exist faces from a predefined dataset of faces in a given 
video/image 
Viola-Jones face detection requires only a gray-scale version of an image
"""
mayim_gray = cv2.cvtColor(mayim_as_img,cv2.COLOR_BGR2GRAY)
show_image(mayim_gray)

# choose classifier and training set
face_classifier = \
    cv2.CascadeClassifier(cv2.data.haarcascades
                          + 'haarcascade_frontalface_default.xml')

face = face_classifier.detectMultiScale(mayim_gray)
cast_gray = cv2.cvtColor(cast_image,cv2.COLOR_BGR2GRAY)


for(_x,_y,_w,_h) in face:
    cv2.rectangle(mayim_as_img,
                  (_x,_y), # upper-left corner
                  (_x+_w,_y+_h), # lower-right corner
                  (0,255,0),
                  10)

show_image(mayim_as_img)


print(type(mayim_as_img))
print(mayim_as_img.shape)
# OpenCV reads image files according the following channels: Blue, Green, Red
