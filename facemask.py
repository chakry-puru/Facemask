import numpy as np, cv2

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
mouth_cascade= cv2.CascadeClassifier ('mouth.xml')
nose_cascade=cv2.CascadeClassifier ('Nose.xml')
while True:
	ret, frame= cap.read()
	fgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(fgray,1.5,5)
	#mask=mask_cascade.detectMultiScale(fgray,1.5,5)
	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		roi_gray=fgray[y:y+h, x:x+w]
		roi_color=frame[y:y+h, x:x+w]
		mmasks= mouth_cascade.detectMultiScale(roi_gray, 1.8, 20)
		nmasks= nose_cascade.detectMultiScale(roi_gray, 1.8, 20)
		if len(mmasks)==0 & len(nmasks)==0:
			cv2.putText(frame,'mask on',(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		elif len(nmasks)!=0 or len(mmasks)!=0:
			cv2.putText(frame,'mask off',(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()