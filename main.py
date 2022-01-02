import  cv2
import idlelib
import datetime
video = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

while(True):

    # Capture frame-by-frame
    ret,frame = video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    eyes=eye_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,0),4)
    for(a,b,c,d) in eyes:
            cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),4)




    cv2.imshow('frame',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


video.release(0)
cv2.destroyAllWindows()