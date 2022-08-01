import cv2
import dlib

cap = cv2.VideoCapture(0)

cnn_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cnn_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            #x = face_landmarks.part(n).x
            #y = face_landmarks.part(n).y
            mouth1x=face_landmarks.part(49).x
            mouth2x=face_landmarks.part(55).x
            smth1=face_landmarks.part(17).y
            smth2=face_landmarks.part(1).y
            smth=smth2-smth1
            #print(smth)
            #print(mouth2x-mouth1x)
            #and cv2.waitKey(33) == ord('a')))
            thex=smth*1.2
            if(mouth2x-mouth1x)>thex:
                #result, image = cap.read()
                #cv2.imshow("image", image)
                #cv2.imwrite("green1.jpeg", image)
                #cv2.imwrite("")
                cv2.putText(frame, "smile", (int(face_landmarks.part(28).x), int(face_landmarks.part(28).y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                
              


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()