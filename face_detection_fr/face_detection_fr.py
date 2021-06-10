import cv2
import face_recognition

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        face_landmarks_list = face_recognition.face_landmarks(frame)
        print(face_landmarks_list)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
