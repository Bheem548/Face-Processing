import cv2
import face_recognition


capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        rect = face_recognition.face_locations(frame, 0, 'hog')
        for face in rect:
            top,right,bottom,left = face
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()