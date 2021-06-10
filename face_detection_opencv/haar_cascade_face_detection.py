import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(img, title, pos):
    img = img[:, :, ::-1]
    plt.subplot(3,2, pos)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def draw_rect(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return img
    


img = cv2.imread('images/scientists.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cas_fa = cv2.CascadeClassifier('Face Processing/cascade_classifiers/haarcascade_frontalface_alt.xml')
cas_fd = cv2.CascadeClassifier('Face Processing/cascade_classifiers/haarcascade_frontalface_default.xml')

fa_img = cas_fa.detectMultiScale(gray)
fd_img = cas_fd.detectMultiScale(gray)

ret, fa_faces = cv2.face.getFacesHAAR(img, 'Face Processing/cascade_classifiers/haarcascade_frontalface_alt.xml')
ret, fd_faces = cv2.face.getFacesHAAR(img, 'Face Processing/cascade_classifiers/haarcascade_frontalface_default.xml')


fa_rect_img = draw_rect(img.copy(), fa_img)
fd_rect_img = draw_rect(img.copy(), fd_img)

fa_get_haar = draw_rect(img.copy(), np.squeeze(fa_faces))
fd_get_haar = draw_rect(img.copy(), np.squeeze(fd_faces))

show_with_matplotlib(img, 'Normal Image', 1)
show_with_matplotlib(fa_rect_img, 'Frontalface_alt.xml : '+ str(len(fa_img)), 3)
show_with_matplotlib(fd_rect_img, 'Frontalface_default.xml : ' + str(len(fd_img)), 4)

show_with_matplotlib(fa_get_haar,'FrontalFace_alt, getFacesHAAR() : '+str(len(fa_faces)),5)
show_with_matplotlib(fd_get_haar,'Frontalface_default, getFacesHAAR() : '+ str(len(fd_faces)),6)
plt.show()
