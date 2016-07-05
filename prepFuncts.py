import numpy as np
import cv2
#from matplotlib import pyplot as plt
from numpy import uint8,zeros,shape,ones,copy,array,float32

def createVectorEasy(img):
    try:
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img1=cv2.resize(img1,(100,100))
        
        #cut face
        fc=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        eyePair=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
        #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
        #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
        nose=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
        #mouth=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
   
        f = fc.detectMultiScale(img1,1.3,5)#return list of find faces
        for (x,y,w,h) in f:
            #cut face
            faceIm=img1[y:y+h,x:x+w]
            
            #find eyes
            eye1 = eyePair.detectMultiScale(faceIm) 
            if len(eye1)!=1:
                pass
                #print("too many eyes pair")
                #raise(myErrors)
            (ex,ey,ew,eh)= eye1[0]
            #eyeIm=faceIm[ey:ey+eh,ex:ex+ew]
        
            #find nose
            nose1=nose.detectMultiScale(faceIm)
            if len(nose1)>1:
                pass
                #print("too many nouses")
                #raise(myErrors)
            #print("len(nose1)=")
            #print(len(nose1))
            (nx,ny,nw,nh) = nose1[0]
            #nouseIm=faceIm[ny:ny+nh,nx:nx+nw]

            #cut new face 
            outFace=img1[y:y+h,x:x+w]
            #then cut it agein
            outFace=outFace[ey:ny+nh,ex:ex+ew]
            #for eye only
            #outFace=outFace[ey:ey+eh,ex:ex+ew]
            outFace=cv2.resize(outFace,(100,50))
        return(outFace)
    except:
        return(None)

def createVectorWithHoG(img):
    try:
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img1=cv2.resize(img1,(100,100))
        #img3=copy(img1)
        # Initiate SIFT detector
        orb = cv2.ORB()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1,None)
        #print("count Keypoints=%s"%len(kp1))
        
        img2=ones(shape(img1))*200
        ##print("shape(img1)=")
        ##print(shape(img1))
        ##print("shape(img2)=")
        ##print(shape(img2))
    
        for xy in kp1:
            img2[xy.pt[1]+1,xy.pt[0]]=0
            img2[xy.pt[1],xy.pt[0]+1]=0
            img2[xy.pt[1]+1,xy.pt[0]+1]=0
            img2[xy.pt[1]-1,xy.pt[0]]=0
            img2[xy.pt[1],xy.pt[0]-1]=0
            img2[xy.pt[1]-1,xy.pt[0]-1]=0
            img2[xy.pt[1],xy.pt[0]]=0
            #img2[xy.pt[::-1]]=254
    
        #cut face
        fc=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
        eyePair=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
        #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
        #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
        #nose=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
        #mouth=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
   
        f = fc.detectMultiScale(img1,1.3,5)#return list of find faces
        for (x,y,w,h) in f:
            #cut face
            faceIm=img1[y:y+h,x:x+w]
            
            #find eyes
            eye1 = eyePair.detectMultiScale(faceIm) 
            if len(eye1)!=1:
                pass
                #print("too many eyes pair")
                #raise(myErrors)
            (ex,ey,ew,eh)= eye1[0]
            #eyeIm=faceIm[ey:ey+eh,ex:ex+ew]
        
            #find nose
            '''
            nose1=nose.detectMultiScale(faceIm)
            if len(nose1)>1:
                print("too many nouses")
                #raise(myErrors)
            print("len(nose1)=")
            print(len(nose1))
            (nx,ny,nw,nh) = nose1[0]
            #nouseIm=faceIm[ny:ny+nh,nx:nx+nw]
            '''

            #cut new face 
            outFace=img2[y:y+h,x:x+w]
            #then cut it agein
            #outFace=outFace[ey:ny+nh,ex:ex+ew]
            #for eye only
            outFace=outFace[ey:ey+eh,ex:ex+ew]
            outFace=cv2.resize(outFace,(100,50))
        return(outFace)
    except:
        return(None)
        

def findKeypoints(imgName):
    '''imgName string of image or
    gray image array (from cv2.imread)
    '''
    if type(imgName)==str:
        img1 = cv2.imread(imgName,0)          # queryImage
    else:
        img1=imgName.astype(uint8)

    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    print("count Keypoints=%s"%len(kp1))
    
    img3  = cv2.drawKeypoints(img1,kp1,color=(0,255,0), flags=0)
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    #return(img3)
    return(kp1)

def match(imgNameQuery,imgNameQueryInScene):
    img1 = cv2.imread(imgNameQuery,0) #queryImage
    img2 = cv2.imread(imgNameQueryInScene,0)
    
    
    
    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    
    # Match descriptors.
    matches = bf.match(des1,des2)    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    #only 10 best results
    kpTmp = [kp2[i.trainIdx] for i in matches[:5]]
    #print(dir(i))
    #draw image and keypints kpTmp
    img3  = cv2.drawKeypoints(img2,kpTmp,color=(0,255,0), flags=0)
    
    #img3  = cv2.drawKeypoints(img1,kp1,color=(0,255,0), flags=0)
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    return(img3)
