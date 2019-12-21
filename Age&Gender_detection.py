import cv2


video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('style.xml')
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

def video_detector(age_net,gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX #Type of font
    while True:
        check,frame = video_capture.read()
        frame=cv2.flip(frame,1)
        #converted our Webcam feed to Grayscale.**< Most of the operations in OpenCV are done in grayscale>**
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2,10,cv2.CASCADE_SCALE_IMAGE,(30,30))
        print(check)
        print(frame)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #Make rectangle on face
            #print(x,y,w,h)#print The coordinates of face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)
            # Get Face as Matrix and copy it
            face_img = frame[y:y + h, h:h + w].copy()
            print(face_img)
            blob=cv2.dnn.blobFromImage(face_img,1,(244,244),MODEL_MEAN_VALUES,swapRB=True)#**
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        key=cv2.waitKey(1)
        if key == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    age_net, gender_net = load_caffe_models()  # load caffe models (age & gender)
    video_detector(age_net, gender_net)  # prediction age & gender

if __name__ == "__main__":
    main()



