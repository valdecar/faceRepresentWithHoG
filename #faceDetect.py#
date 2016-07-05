from numpy import *
import cv2
from scipy.fftpack import fft,ifft,fft2,ifft2
from skimage import filter
#import matplotlib.pyplot as plt
myErrors=StandardError()
def prepEye(img,m=6):
    '''img is colored image of eye
    '''
    img = filt(img)
    e=fftElips(img,m)
    return(e)
def filt1(img,m=2):
    '''
    it is best I think
    '''
    #imm=fftElips(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mm = mean(img)
    img1= copy(img)
    img2 = copy(img)
    img1[img1<mm]=0
    img2[img2>mm]=0
    img11=filter.gaussian_filter(img1,sigma=m)
    img21=filter.gaussian_filter(img2,sigma=m)
    imgNew=zeros(shape(img))
    for i in range(shape(img)[0]):
        for j in range(shape(img)[1]):
            if (img1[i,j]==0):
                imgNew[i,j]=img11[i,j]
            if (img2[i,j]==0):
                imgNew[i,j]=img21[i,j]
    #imm=filter.gaussian_filter(img,sigma=m)
    return(imgNew,img1,img11,img2,img21)#imm

def filt(img,m=2):
    '''
    it is best I think
    '''
    #imm=fftElips(img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imm=filter.gaussian_filter(img,sigma=m)
    return(imm)
def sub(e):
    f,a=plt.subplots(3)
    a[0].imshow(e[0])
    a[1].imshow(e[2])
    a[2].imshow(e[3])
    plt.show()
def eyes(s):
    ''' s from pikle.load file finalDataAND_m1T_AND_mean
    file = open()
    pikle.load(file)
    file.close()
    '''
    eyes=[]
    for i in range(len(s[0][2])):
        try:
            e=findFaceColor(s[0][2][i])[1][0]
            #ee=f(e) or that
            ee=filt(e)
            ee1=filt1(e)
            eyes.append([ee,ee1])
        except IndexError:
            print("eyes no found\n")
    return(eyes)
def f(img):
    '''traing find ellips 
    '''
    img1=fftElips(img)
    imm = filter.gaussian_filter(img1,sigma=2)
    imm=ones(shape(imm))*255-imm
    imm1 = copy(imm)
    immDown=copy(imm1)
    immUp=copy(imm1)
    immDown[immDown<(mean(immDown))]=0
    immUp[immUp>mean(immUp)]=0
    #return(imm)
    imm[imm<mean(imm)]=0
    m = mean(imm[imm>0])
    imm[imm<m]=0
    return(imm,imm1,immDown,immUp)
def fftElips(eye,m=1):
    '''Return foure frequencis and repear image
    eye is green image
    '''
    #eye = findFaceColor(img)[1]
    #eye=cv2.cvtColor(eye[0],cv2.COLOR_BGR2GRAY)
    #eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    y=fft2(eye)
    y[m:-m,m:-m]=0
    #return(y)
    yy=ifft2(y)
    return([y,real(yy)])
    yy[real(yy)>100]=0
    hist=cv2.calcHist([real(yy).astype(uint8)],[0],None,[256],[0,256])
    hist=concatenate(hist)
    return(real(yy),hist)
def eyeElips(img):
    
    f2=MMCC(img)
    aa= copy(img)
    aa[f2(aa)<0]=[0,0,0]
    return(aa)
    f21=MMCC(aa,2)
    aaa=copy(aa)
    aaa[f21(aaa)<0]=[0,0,0]
    f211=MMCC(aaa,3)
    ab=copy(aaa)
    ab[f211(ab)<0]=[0,0,0]
    return(ab)
def MMCC(img,m=1):
    ''' m=1  eignvector for max engval;m=3 eignvector for min engval
    
    '''
    img1 = img.astype(float32)
    c=cov(img1.reshape((-1,3)).T)
    evl,envt=linalg.eigh(c)
    maxE=envt.T[-m]
    E=mean(img1.reshape((-1,3)),axis=0)
    d=-sum(maxE*E)
    f=lambda x:sum(maxE*x)+d
    f2=lambda x:sum(maxE*x,axis=2)+d#axis 2 for sum in img1 with shape (w,h,3)
    return(f2)#return hyperplane
    img2 =copy(img1)
    if sign(mean(maxE))>0:
        img2[f2(img2)<0]=[0,0,0]
    else:
        img2[f2(img2)>0]=[0,0,0]
    print(maxE)
    #g  = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ##g1=cv2.resize(g,(50,50))
    return(img2)#img2

def findExtr(img,n=3):
    '''This funct solve fft of histogram and filtering phase spece use first n 
    transition throuth zero (ie find zeros of diff(abs(fft(hist))) and save only maximum
    frequencis ie z[In:-In]=0)
    Then restore y (using ifft) 
    
    '''
    #find histogram
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([img.astype(uint8)],[0],None,[256],[0,256])
    hist=concatenate(hist)
    z=fft(hist)
    s=[]
    #find transition throuth zero with - to +
    dy = diff(abs(z))
    for i in range(len(dy))[1:]:
        if sign(dy[i])+sign(dy[i-1])==0:
            if(sign(dy[i-1])<0):
                s.append(i)
                if len(s)>=n:break
    print(s)#s contain x : f(x)=0 

    #filtering data use only first n frequency component
    z[s[-1]:-s[-1]]=0
    #restore data
    y=real(ifft(z))
    return((y,hist))

def findFaceColorEasy(img,grey=False):
    '''find face between eyes and mounth
    input 
    img is colored image with men face
    faceCutType=0 then cut between eyes and nouse
    faceCutType=1 then cut between eyes and mounth
    nouse and eyes filtered separately
    output
    image float32 and /mean(img) for norming and grey
    '''
    img=copy(img)
    if not grey:
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img1=img

    fc=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eyePair=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
    #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
    nose=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
    mouth=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
    #img = cv2.imread('12.1.jpg')
    #g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(img1,1.3,5)#return list of find faces
    outImg = []

    for (x,y,w,h) in f:
        outImg.append(img[y:y+h,x:x+w])
        faceIm=img[y:y+h,x:x+w]
        if not grey:
            faceImGr=cv2.cvtColor(faceIm,cv2.COLOR_BGR2GRAY)
        else:
            faceImGr=copy(faceIm)

        eye1 = eyePair.detectMultiScale(faceImGr) 
        if len(eye1)!=1:
            print("too many eyes pair")
            raise(myErrors)
        (ex,ey,ew,eh)= eye1[0]
        eyeIm=faceIm[ey:ey+eh,ex:ex+ew]

        ##CUT FACE BETWEEN EYES AND NOUSE 
        faceImGr=faceImGr[ey:]
        faceImGr=faceImGr[:,ex:ex+ew]
        #final prepear
        #faceImGr=faceImGr.astype(float32)
        ##faceImGr=faceImGr/float(mean(faceImGr))
        #eyeIm=cv2.cvtColor(eyeIm,cv2.COLOR_BGR2GRAY)
        #faceImGr=cv2.resize(faceImGr,(50,50))
        return(faceImGr,eyeIm)
    

def findFaceColor(img):
    '''find face,eyes,nose,mouth
    '''
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fc=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
    #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
    nose=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
    mouth=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
    #img = cv2.imread('12.1.jpg')
    #g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(img1,1.3,5)#return list of find faces
    outImg = []
    outEye =[]
    outNose=[]
    outMouth=[]
    for (x,y,w,h) in f:
        outImg.append(img[y:y+h,x:x+w])
        faceIm=img[y:y+h,x:x+w]
        faceImGr=cv2.cvtColor(faceIm,cv2.COLOR_BGR2GRAY)
        eye1 = eye.detectMultiScale(faceImGr) 
        nose1=nose.detectMultiScale(faceImGr)
        mouth1=mouth.detectMultiScale(faceImGr)
        for (ex,ey,ew,eh) in eye1:
            eyeIm=faceIm[ey:ey+eh,ex:ex+ew]
            outEye.append(eyeIm)  
        for (nx,ny,nw,nh) in nose1:
            nouseIm=faceIm[ny:ny+nh,nx:nx+nw]
            outNose.append(nouseIm)
        for (mx,my,mw,mh) in mouth1:
            mouthIm=faceIm[my:my+mh,mx:mx+mw]
            outMouth.append(mouthIm)
    return((outImg,outEye,outNose,outMouth))

def findFaceColor1(img,faceCutType=0,grey=False):
    '''find face between eyes and mounth
    input 
    img is colored image with men face
    faceCutType=0 then cut between eyes and nouse
    faceCutType=1 then cut between eyes and mounth
    nouse and eyes filtered separately
    output
    image float32 and /mean(img) for norming and grey
    '''
    img=copy(img)
    if not grey:
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img1=img

    fc=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eyePair=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_big.xml')
    eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
    #eye=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_righteye_2splits.xml')
    nose=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
    mouth=cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')
    #img = cv2.imread('12.1.jpg')
    #g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f = fc.detectMultiScale(img1,1.3,5)#return list of find faces
    outImg = []
    outEye =[]
    outNose=[]
    outMouth=[]
    for (x,y,w,h) in f:
        outImg.append(img[y:y+h,x:x+w])
        faceIm=img[y:y+h,x:x+w]
        if not grey:
            faceImGr=cv2.cvtColor(faceIm,cv2.COLOR_BGR2GRAY)
        else:
            faceImGr=copy(faceIm)

        eye1 = eyePair.detectMultiScale(faceImGr) 
        if len(eye1)>1:
            print("too many eyes pair")
            raise(myErrors)
        #find right eye
        eye2=eye.detectMultiScale(faceImGr)
        print('len(eye2)=')
        print(len(eye2))
        #find left eye
        (elx,ely,elw,elh)= eye2[0]
        eyeLeftIm=faceImGr[ely:ely+elh,elx:elx+elw]
        #filter left eye
        eyeLeftIm=fftElips(eyeLeftIm,m=2)[1]
        eyeLeftIm[eyeLeftIm<mean(eyeLeftIm)]=0
        #insert new left eye
        faceImGr[ely:ely+elh,elx:elx+elw]=eyeLeftIm

        #find right eye
        (erx,ery,erw,erh)= eye2[1]
        eyeRightIm=faceImGr[ery:ery+erh,erx:erx+erw]
        #filter right eye
        eyeRightIm=fftElips(eyeRightIm,m=2)[1]
        eyeRightIm[eyeRightIm<mean(eyeRightIm)]=0
        #insert new right eye
        faceImGr[ery:ery+erh,erx:erx+erw]=eyeRightIm

        nose1=nose.detectMultiScale(faceImGr)
        if len(nose1)>1:
                print("too many nouses")
                raise(myErrors)
        print("len(nose1)=")
        print(len(nose1))
        (nx,ny,nw,nh) = nose1[0]
        nouseIm=faceImGr[ny:ny+nh,nx:nx+nw]
        #filter nose
        nouseIm[nouseIm<=mean(nouseIm[nouseIm<mean(nouseIm)])]=0
        #insert new nose
        faceImGr[ny:ny+nh,nx:nx+nw]=nouseIm

        mouth1=mouth.detectMultiScale(faceImGr)
        print('len(mouth1)=')
        print(len(mouth1))
            
        (ex,ey,ew,eh)= eye1[0]
        eyeIm=faceIm[ey:ey+eh,ex:ex+ew]
        #eyeImGr=cv2.cvtColor(eyeIm,cv2.COLOR_BGR2GRAY)
        #eye2=eye.detectMultiScale(eyeImGr)
        #print('len(eye2)=')
        #print(len(eye2))
        if faceCutType==1:
            ##CUT FACE BETWEEN EYES AND MOUTH
            #delete eyes from mouth1
            mouth2=[]
            for m in mouth1:
                mx=m[0]
                my=m[1]
                mw=m[2]
                mh=m[3]
                if my<=ey+eh:
                    print("removed element")
                else:
                    mouth2.append(m)                
                    print('len(mouth2)=')
            print(len(mouth2))       
            if len(mouth2)>1:
                print("too many mouths")
                raise(myErrors)
            (mx,my,mw,mh) = mouth2[0]
            mouthIm=faceImGr[my:my+mh,mx:mx+mw]
            #find lips
            sx = nd.sobel(mouthIm,axis=0)
            sy = nd.sobel(mouthIm,axis=1)
            Gr = hypot(sx,sy)
            Gr[Gr<mean(Gr[Gr>mean(Gr)])]=0
            
            #cut image between eyes and mouth
            faceImGr=faceImGr[ey:my+mh]
            faceImGr=faceImGr[:,ex:ex+ew]

            return(faceImGr,mouthIm)
        else:
            ##CUT FACE BETWEEN EYES AND NOUSE           
            #cut image between eyes and nose
            faceImGr=faceImGr[ey:ny+nh]
            faceImGr=faceImGr[:,ex:ex+ew]
            #final prepear
            faceImGr=faceImGr.astype(float32)
            ##faceImGr=faceImGr/float(mean(faceImGr))
            #eyeIm=cv2.cvtColor(eyeIm,cv2.COLOR_BGR2GRAY)
            #faceImGr=cv2.resize(faceImGr,(50,50))
            return(faceImGr,nouseIm,eyeIm,eyeLeftIm,eyeRightIm)
        
