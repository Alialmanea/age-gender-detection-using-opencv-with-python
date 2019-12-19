
import cv2


video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('style.xml')
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def video_detector(gender_net):
    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        check, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 10)
        if (len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))
        print(check)
        print(frame)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #Make rectangle on face
            print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0,0), 3)
            # Get Face
            face_img = gray[y:y + h, h:h + w].copy()
            blob=cv2.dnn.blobFromImage(face_img,1,(277,277),MODEL_MEAN_VALUES,swapRB=False)
            #predicct gender
        cv2.imshow('Capturing', frame)
        key=cv2.waitKey(1)
        if key == 27:
            break
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    video_detector(gender_net)


if __name__ == "__main__":
    main()

