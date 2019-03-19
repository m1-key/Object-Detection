#@Author-Abhay Katheria


#IMPORTING THE LIBRARIES
import cv2 , time

#DECLARING FIRST FRAME AS NONE
firstf= None

#STARTING THE VIDEO CAPTURE CV2.VIDEOCAPTURE TAKES THE INPUT THE SERIAL NO. OF WEBCAM 0 IF WE ARE USING INTERNAL
video = cv2.VideoCapture(0)

#INITIATING WHILE LOOP WHICH CAN BREAK ONLY FROM INSIDE
while True :
    
    #READING THE FRAME
    check,frame= video.read()

    #CV2 WORKS BEST FOR GRAY SCALE IMAGES THATS WHY WE CONVERT EACH FRAME INTO GRAYSCALE
    greyscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    

    #APPLYING GAUSIAN BLUR GAUSSIAN BLUR SMOOTHENS THE IMAGE AND THUS REDUCES THE NOISE
    greyscale = cv2.GaussianBlur(greyscale,(21,21),0)

    # THE MOST IMPORTANT STEP SETTING THE FIRST FRAME
    if firstf is None :
        firstf = greyscale
        continue
    
    #FINDING THE DIFFERENCE B/W CURRENT FRAME AND FIRST FRAME
    delta = cv2.absdiff(firstf,greyscale)
    
    #FINDING THE THRESHHOLD FRAME AS WE CAN DRAW COUNTOURS ON THRESH FRAME ONLY 
 	#THE SECOND ARGUMENT IN THE THRESHHOLD FUNCTION IS THE DIFFERENCE BELOW WHICH WE WILL NOT CONSIDER MOTION
 	#IT IS A HYPERPARAMETER AND CAN BE FINE TUNED AS PER ROOM CONDITIONS
    thresh = cv2.threshhold(delta,30,255,cv2.THRESH_BINARY)[1]
    thresh= cv2.dilate(thresh,None,iterations = 2)
    

    #FINDING CONTOURS
    (_,cnts,_)=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    

    #FINDING CONTOURS HAVING AREA GREATER THAN AREA 1000 AND MAKING RECTANGLE AROUND THEM	
    for contour in cnts :
    	if cv2.contourArea(contour) < 1000:
    		continue
    	[x,y,h,w] = cv2.boundingRect(contour)
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    print(frame)

    print(delta)


    cv2.imshow("cap",frame)
    cv2.imshow("delta",delta)
    cv2.imshow("threshhold",thresh)

    key = cv2.waitKey(2)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()


