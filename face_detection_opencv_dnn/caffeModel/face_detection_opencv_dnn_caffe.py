import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_with_matplotlib(img, title, pos):
    img = img[:, :, ::-1]
    plt.subplot(1, 2, pos)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

img = cv2.imread('images/scientists.png')

( h,w) = img.shape[:2]

net = cv2.dnn.readNetFromCaffe(
    "Face Processing/face_detection_opencv_dnn/caffe_model/deploy.prototxt",
    "Face Processing/face_detection_opencv_dnn/caffe_model/res10_300x300_ssd_iter_140000_fp16.caffemodel")


blob = cv2.dnn.blobFromImage(img, 1.0, (img.shape[:2]), [104., 117., 123.], False, False)
net.setInput(blob)
detections = net.forward()

detected_faces = 0

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence>0.3:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        cv2.rectangle(img,(startX,startY),(endX,endY),(0,255,0),3)
        

show_with_matplotlib(img, 'Normal Image', 1)

plt.show()
