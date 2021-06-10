import cv2

capture = cv2.VideoCapture(0)

cascade_classifier = cv2.CascadeClassifier(
    "Face Processing\cascade_classifiers\haarcascade_frontalface_alt.xml")
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
