import cv2

cap = cv2.VideoCapture(0)
face_cascede = cv2.CascadeClassifier("frontal_face.xml")
while True:
    ret,frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascede.detectMultiScale(gray,1.4,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x + w , y + h),(0,0,255),thickness=5)
    cv2.imshow("Face Detection",frame)    
        
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()    