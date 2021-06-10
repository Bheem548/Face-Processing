import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(img, title, pos):
    img = img[:, :, ::-1]
    plt.subplot(1, 2, pos)
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')

def draw_rect(img, faces):
    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)
    return img
img = cv2.imread('images/scientists.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()

rect1 = detector(gray, 2)

img_rect = draw_rect(img.copy(),rect1)
show_with_matplotlib(img, 'Normal Image', 1)
show_with_matplotlib(img_rect, 'With 0 upsample : '+ str(len(rect1)), 2)

plt.show()