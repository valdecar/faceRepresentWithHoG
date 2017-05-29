'''
DESCRIPTION:
 This file contain some methods for face
extraction as vector in features space.
 Some of that methods trying to extract
face ellips.

FUNCTIONS:

prep_eye(img, m=6)

# filters
face_extract_gauss_2(img, m=2)
face_extract_gauss(img, m=2)
face_extract_mean(img)

# elips excraction
elips_fft(eye, m=1)
elips_mmcc(img)

# methods
MMCC(img, m=1)
find_extr_of_hist(img, n=3)

# face as vector
find_face(img)
find_face_color_eyes_nouse(img,
                           faceCutType=0,
                           grey=False)
'''

from numpy import mean, copy, zeros, shape
from numpy import ones, real, uint8, float32
from numpy import concatenate, cov, linalg
from numpy import diff, sign, sum

import cv2
from scipy.fftpack import fft, ifft, fft2, ifft2
from skimage import filters as filter

import faceObjectsSearchers as fos

import matplotlib.pyplot as plt


def prep_eye(img, m=6):
    '''
    DESCRIPTION:
    
    INPUT:
    img is colored image of eye
    '''

    img = face_extract_gauss(img)
    e = elips_fft(img, m)
    return(e)


def face_extract_gauss_2(img, m=2):
    '''
    DESCRIPTION:
    try to extract face
    elipse from image.
    
    INPUT:
    img is colored image.
    '''
    # imm=fftElips(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mm = mean(img)
    img1 = copy(img)
    img2 = copy(img)
    img1[img1 < mm] = 0
    img2[img2 > mm] = 0
    img11 = filter.gaussian_filter(img1, sigma=m)
    img21 = filter.gaussian_filter(img2, sigma=m)
    imgNew = zeros(shape(img))
    for i in range(shape(img)[0]):
        for j in range(shape(img)[1]):
            if (img1[i, j] == 0):
                imgNew[i, j] = img11[i, j]
            if (img2[i, j] == 0):
                imgNew[i, j] = img21[i, j]

    # imm = filter.gaussian_filter(img, sigma=m)
    return(imgNew, img1, img11, img2, img21)  # imm


def face_extract_gauss(img, m=2):
    '''
    DESCRIPTION:
    try to extract face
    elipse from image.
    
    it is best I think
    
    INPUT:
    img is grey image.
    '''

    # imm = fftElips(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imm = filter.gaussian_filter(img, sigma=m)
    return(imm)


def face_extract_mean(img):
    '''
    DESCRIPTION:
    Try to find face ellips.
    
    '''
    img1 = elips_fft(img)
    imm = filter.gaussian_filter(img1, sigma=2)
    imm = ones(shape(imm))*255 - imm
    imm1 = copy(imm)
    immDown = copy(imm1)
    immUp = copy(imm1)
    immDown[immDown < (mean(immDown))] = 0
    immUp[immUp > mean(immUp)] = 0
    # return(imm)
    imm[imm < mean(imm)] = 0
    m = mean(imm[imm > 0])
    imm[imm < m] = 0
    return(imm, imm1, immDown, immUp)


def elips_fft(eye, m=1):
    '''
    DESCRIPTION:
    Filter image eye by fourier m hugest
    frequencis in frequencis space.    
    
    INPUT:
    eye is grey image

    OUTPUT:
    y - frequencis 
    yy - filtred image
    '''

    # eye = findFaceColor(img)[1]
    # eye=cv2.cvtColor(eye[0], cv2.COLOR_BGR2GRAY)
    # eye=cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    y = fft2(eye)

    # cut frequencis form
    # low to huge
    y[m:-m, m:-m] = 0
    
    yy = ifft2(y)

    return([y, real(yy)])

    yy[real(yy) > 100] = 0
    hist = cv2.calcHist([real(yy).astype(uint8)],
                        [0], None, [256], [0, 256])
    hist = concatenate(hist)
    return(real(yy), hist)


def elips_mmcc(img):
    '''
    DESCRIPTION:
    Extract elips using
    main components method.
    i.e. engspace in colors.
    '''
    f2 = MMCC(img)
    aa = copy(img)
    aa[f2(aa) < 0] = [0, 0, 0]

    return(aa)

    f21 = MMCC(aa, 2)
    aaa = copy(aa)
    aaa[f21(aaa) < 0] = [0, 0, 0]
    f211 = MMCC(aaa, 3)
    ab = copy(aaa)
    ab[f211(ab) < 0] = [0, 0, 0]
    return(ab)


def MMCC(img, m=1):
    ''' 
    DESCRIPTION:
    Main component method used for
    extract face elipse.
    m=1  eignvector for max engval;m=3 eignvector for min engval
    
    OUTPUT:
    f2 - hyperplane that classify
         face from other.
    '''
    img1 = img.astype(float32)
    c = cov(img1.reshape((-1, 3)).T)
    evl, envt = linalg.eigh(c)
    maxE = envt.T[-m]

    # see test_base_math for comments
    E = mean(img1.reshape((-1, 3)), axis=0)
    
    d = -sum(maxE * E)

    f = lambda x: sum(maxE*x)+d
    f2 = lambda x: sum(maxE*x, axis=2)+d  # axis 2 for sum in img1 with shape (w,h,3)
    
    return(f2)  # return hyperplane
    
    img2 = copy(img1)
    if sign(mean(maxE)) > 0:
        img2[f2(img2) < 0] = [0, 0, 0]
    else:
        img2[f2(img2) > 0] = [0, 0, 0]
    print(maxE)
    # g  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # g1 = cv2.resize(g, (50, 50))
    return(img2)  # img2


def find_extr_of_hist(img, n=3):

    '''
    DESCRIPTION:
    This funct solve fft of histogram and
    filtering phase spece use first n
    transition throuth zero (ie find zeros
    of diff(abs(fft(hist))) and save only
    maximum frequencis ie z[In:-In] = 0)
    Then restore y (using ifft).    
    '''

    # find histogram
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img.astype(uint8)], [0],
                        None, [256], [0, 256])
    hist = concatenate(hist)
    z = fft(hist)
    s = []

    # find transition throuth zero with - to +
    dy = diff(abs(z))
    for i in range(len(dy))[1:]:
        if sign(dy[i])+sign(dy[i-1]) == 0:
            if(sign(dy[i-1]) < 0):
                s.append(i)
                if len(s) >= n:
                    break
    print(s)  # s contain x : f(x) = 0

    # filtering data use only first n frequency component
    z[s[-1]:-s[-1]] = 0
    # restore data
    y = real(ifft(z))
    return((y, hist))


def find_face(img):
    '''
    DESCRIPTION:
    Find face, eyes, nose, mouth.

    INPUT:
    img - colored image.
    '''

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fc = fos.get_face()
    eye = fos.get_eye_pair()
    nose = fos.get_nose()
    mouth = fos.get_mouth()
    
    f = fc.detectMultiScale(img1, 1.3, 5)  # return list of find faces
    outImg = []
    outEye = []
    outNose = []
    outMouth = []
    for (x, y, w, h) in f:
        outImg.append(img[y:y+h, x:x+w])
        faceIm = img[y:y+h, x:x+w]

        faceImGr = cv2.cvtColor(faceIm, cv2.COLOR_BGR2GRAY)

        eye1 = eye.detectMultiScale(faceImGr)
        nose1 = nose.detectMultiScale(faceImGr)
        mouth1 = mouth.detectMultiScale(faceImGr)

        for (ex, ey, ew, eh) in eye1:
            eyeIm = faceIm[ey:ey+eh, ex:ex+ew]
            outEye.append(eyeIm)
  
        for (nx, ny, nw, nh) in nose1:
            nouseIm = faceIm[ny:ny+nh, nx:nx+nw]
            outNose.append(nouseIm)

        for (mx, my, mw, mh) in mouth1:
            mouthIm = faceIm[my:my+mh, mx:mx+mw]
            outMouth.append(mouthIm)
    return((outImg, outEye, outNose, outMouth))


def find_face_color_eyes_nose(img, faceCutType=0, grey=False):
    '''
    DESCRIPTION:
    Find face between eyes and nouse.
    
    INPUT:
    img is colored image with men face
    faceCutType=0 then cut between eyes and nouse
    faceCutType=1 then cut between eyes and mounth
    nouse and eyes filtered separately
    output
    image float32 and /mean(img) for norming and grey
    '''

    img = copy(img)
    if not grey:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img1 = img

    eye = fos.get_eye_pair_big()
    eyeL = fos.get_eye_left()
    eyeR = fos.get_eye_right()
    nose = fos.get_nose()
    mouth = fos.get_mouth()

    faceAndBorders = fos.find_face(img1)
    faceIm = faceAndBorders[0]
    (x, y, w, h) = faceAndBorders[1]
    
    faceImGr = faceIm
    # find eyes borders
    (ex, ey, ew, eh) = fos.find_at_face_borders(faceImGr, eye)
    eyeIm = copy(faceIm[ey:ey+eh, ex:ex+ew])
    
    # find left eye
    faceImGr = _find_and_filter_eye(faceImGr, eyeL)

    # find right eye
    faceImGr = _find_and_filter_eye(faceImGr, eyeR)
     
    # find nose
    faceImGr, nc = _find_and_filter_nose(faceImGr, nose)
    (nx, ny, nw, nh) = nc

    # find mounth
    mouth1 = mouth.detectMultiScale(faceImGr)
    print('len(mouth1)=')
    print(len(mouth1))

    if faceCutType == 1:
        
        # CUT FACE BETWEEN EYES AND MOUTH
        # faceImGr, mouthImg = cut_face_between_eye_and_mouth(faceImGr,
        #                                                     (ex, ey, ew, eh))
        print("faceCutType = 1 is not suported now")
        return(None)
    else:
        # CUT FACE BETWEEN EYES AND NOUSE           
        # cut image between eyes and nose
        faceImGr = faceImGr[ey:ny+nh]
        faceImGr = faceImGr[:, ex:ex+ew]

        # final prepear
        faceImGr = faceImGr.astype(float32)
        # faceImGr = faceImGr/float(mean(faceImGr))
        # eyeIm = cv2.cvtColor(eyeIm, cv2.COLOR_BGR2GRAY)
        # faceImGr = cv2.resize(faceImGr, (50, 50))
        return(faceImGr, eyeIm)


def _find_and_filter_eye(faceImGr, eye):
    # find  eye
    (elx, ely, elw, elh) = fos.find_at_face_borders(faceImGr, eye)
    eyeLeftIm = copy(faceImGr[ely:ely+elh, elx:elx+elw])

    # filter eye
    eyeLeftIm = elips_fft(eyeLeftIm, m=2)[1]
    eyeLeftIm[eyeLeftIm < mean(eyeLeftIm)] = 0

    # insert new  eye
    faceImGr[ely:ely+elh, elx:elx+elw] = eyeLeftIm
    return(faceImGr)


def _find_and_filter_nose(faceImGr, nose):
    (nx, ny, nw, nh) = fos.find_at_face_borders(faceImGr, nose)
    nouseIm = copy(faceImGr[ny:ny+nh, nx:nx+nw])

    # filter nose
    nouseIm[nouseIm <= mean(nouseIm[nouseIm < mean(nouseIm)])] = 0

    # insert new nose
    faceImGr[ny:ny+nh, nx:nx+nw] = nouseIm
    return(faceImGr, (nx, ny, nw, nh))


def sub(e):
    '''
    DESCRIPTION:
    Plot all in one frame

    INPUT:
    e is list with 3 images.
    '''

    f, a = plt.subplots(3)
    a[0].imshow(e[0])
    a[1].imshow(e[2])
    a[2].imshow(e[3])
    plt.show()


def eyes(s):
    '''
    DESCRIPTION:
    Find eyes.

    INPUT:
    s from pikle.load file finalDataAND_m1T_AND_mean
    
    EXAMPLE:
    file = open()
    pikle.load(file)
    file.close()
    '''

    eyes = []
    for i in range(len(s[0][2])):
        try:
            e = findFaceColor(s[0][2][i])[1][0]
            # ee=f(e) or that
            ee = filt(e)
            ee1 = filt1(e)
            eyes.append([ee, ee1])
        except IndexError:
            print("eyes no found\n")
    return(eyes)

'''
def cut_face_between_eye_and_mouth(mouth1, faceImGr, ec):
        # delete eyes from mouth1
        mouth2 = []
        for m in mouth1:
            mx = m[0]
            my = m[1]
            mw = m[2]
            mh = m[3]
            if my <= ey + eh:
                print("removed element")
            else:
                mouth2.append(m)                
                print('len(mouth2)=')
        print(len(mouth2))

        if len(mouth2) > 1:
            print("too many mouths")
            raise(myErrors)
        (mx, my, mw, mh) = mouth2[0]
        mouthIm = faceImGr[my:my+mh, mx:mx+mw]

        # find lips
        sx = nd.sobel(mouthIm, axis=0)
        sy = nd.sobel(mouthIm, axis=1)
        Gr = hypot(sx, sy)
        Gr[Gr < mean(Gr[Gr > mean(Gr)])] = 0

        # cut image between eyes and mouth
        (ex, ey, ew, eh) = ec
        faceImGr = faceImGr[ey:my+mh]
        faceImGr = faceImGr[:, ex:ex+ew]

        return(faceImGr, mouthIm)
    
'''
