import cv2
from numpy import copy


def find_face(img):

    # init face classifier
    fc = get_face()

    # find face
    f = fc.detectMultiScale(img, 1.3, 5)  # return list of find faces
    (x, y, w, h) = f[0]

    # cut face
    return([img[y:y+h, x:x+w], (x, y, w, h)])


def find_at_face_borders(img, classifier):
    o = classifier.detectMultiScale(img)
    if len(o) != 1:
        pass
        # print("too many eyes pair")
    return(o[0])


def find_at_face_object(faceImg, classifier):
    # find eyes borders
    (ex, ey, ew, eh) = find_at_face_borders(faceImg, classifier)
    return(copy(faceImg[ey:ey+eh, ex:ex+ew]))

    
def get_face():
    fc = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    return(fc)


def get_eye_pair_big():
    eyePair = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    return(eyePair)


def get_eye_pair():
    eye = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
    return(eye)


def get_eye_right():
    eye = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
    return(eye)


def get_eye_left():
    eye = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_lefteye_2splits.xml')
    return(eye)


def get_nose():
    nose = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
    return(nose)


def get_mouth():
    mouth = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
    return(mouth)
