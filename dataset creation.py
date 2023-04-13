import cv2, os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'
sub_data = 'Team'

 

path = os.path.join(datasets, sub_data)#/dataset/Team
if not os.path.isdir(path) : #if this folder is not present
    os.mkdir(path)    #create that folder
(width, height) = (100, 100)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)

count = 1

while count <31:
    print (count)
    (_, im) = webcam.read()
    
    RGBImage = im
    faces = face_cascade.detectMultiScale(RGBImage,1.3,4)#detect face
    for (x,y,w,h) in faces:
        cv2.rectangle (im, (x,y), (x+w,y+h), (255,0,0),2) #draw rectangle around face
        face = RGBImage[y:y + h, x:x + w]
        face_resize = cv2.resize (face,(width, height) )
        cv2.imwrite('%s/%s.png' % (path,count),face_resize)
        count += 1

    cv2.imshow('OpenCv', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()


