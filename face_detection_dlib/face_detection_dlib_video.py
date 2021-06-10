import cv2
import dlib

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
while capture.isOpened():
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret:
        faces = detector(gray, 0)
        for face in faces:
            cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(0,255,0),3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()