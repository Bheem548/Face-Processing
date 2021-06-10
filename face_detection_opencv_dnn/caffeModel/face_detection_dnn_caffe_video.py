import cv2
import numpy as np

capture = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(
    "Face Processing/face_detection_opencv_dnn/caffe_model/deploy.prototxt",
    "Face Processing/face_detection_opencv_dnn/caffe_model/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (frame.shape[:2]), [104., 117., 123.], False, False)
        (h,w) = frame.shape[:2]
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0,i, 3:7] * np.array([w, h, w, h])
                startX,startY,endX,endY = box.astype('int')
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
