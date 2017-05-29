import cv2
from numpy import uint8, shape
from numpy import ones
import faceObjectsSearchers as fos


def create_vector_easy(img):
    '''
    DESCRIPTION:
    Just Find face and use it for 
    vector.

    INPUT:
    img - colored image.

    OUTPUT:
    outFace - image eyes pair.
    '''

    try:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img1 = cv2.resize(img1, (100, 100))
        
        # find face
        faceAndBorders = fos.find_face(img1)
        faceIm = faceAndBorders[0]
        (x, y, w, h) = faceAndBorders[1]

        # find eyes
        eyePair = fos.get_eye_pair_big()
        (ex, ey, ew, eh) = fos.find_at_face_borders(faceIm, eyePair)

        # find nose
        nose = fos.get_nose()
        (nx, ny, nw, nh) = fos.find_at_face_borders(faceIm, nose)
        
        # cut new face
        outFace = img1[y:y+h, x:x+w]

        # then cut it again
        outFace = outFace[ey:ny+nh, ex:ex+ew]

        outFace = cv2.resize(outFace, (100, 50))

        return(outFace)

    except:
        return(None)


def create_vector_with_HoG(img):
    '''
    DESCRIPTION:
    Find keypints by HoG at image,
    then remove all keypints that not in
    face area. 
    It will be result vector.
    '''
    try:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img1 = cv2.resize(img1, (100, 100))
        # img3 = copy(img1)

        # find keypints
        kp1 = find_keypoints(img1)

        # create image with only 
        # keypints
        img2 = ones(shape(img1))*0  # 200

        for xy in kp1:
            img2[int(xy.pt[1]+1), int(xy.pt[0])] = 200
            img2[int(xy.pt[1]), int(xy.pt[0]+1)] = 200
            img2[int(xy.pt[1]+1), int(xy.pt[0]+1)] = 200
            img2[int(xy.pt[1]-1), int(xy.pt[0])] = 200
            img2[int(xy.pt[1]), int(xy.pt[0]-1)] = 200
            img2[int(xy.pt[1]-1), int(xy.pt[0]-1)] = 200
            img2[int(xy.pt[1]), int(xy.pt[0])] = 200
            # img2[xy.pt[::-1]] = 254

        # find face
        faceAndBorders = fos.find_face(img1)
        faceIm = faceAndBorders[0]
        (x, y, w, h) = faceAndBorders[1]

        # find eyes
        eyePair = fos.get_eye_pair_big()
        (ex, ey, ew, eh) = fos.find_at_face_borders(faceIm, eyePair)

        # eyeIm = faceIm[ey:ey+eh, ex:ex+ew]

        # find nose
        # nose=get_nose(faceIm)
        # find_at_face_borders(faceIm, nose)
        # nouseIm=faceIm[ny:ny+nh,nx:nx+nw]

        # cut new face
        outFace = img2[y:y+h, x:x+w]

        # then cut it again
        # outFace = outFace[ey:ny+nh, ex:ex+ew]
        # for eye only
        outFace = outFace[ey:ey+eh, ex:ex+ew]
        outFace = cv2.resize(outFace, (100, 50))

        return(outFace)
    except:
        return(None)


def find_keypoints(imgName):

    '''
    DESCRIPTION:
    Find keypints of image.

    INPUT:
    imgName -  string of image or
               gray image array 
               (from cv2.imread)
    '''

    if type(imgName) == str:
        img1 = cv2.imread(imgName, 0)  # queryImage
    else:
        img1 = imgName.astype(uint8)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    
    # print("count Keypoints=%s" % len(kp1))
    
    # img3 = cv2.drawKeypoints(img1, kp1, color=(0, 255, 0), flags=0)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # return(img3)
    return(kp1)


def match(imgNameQuery, imgNameQueryInScene):
    '''
    DESCRIPTION:
    Match two images in HoG points space, 
    then show 10 best matched points at
    imgNameQueryInScene.

    INPUT:
    imgNameQuery - image which need to find
    imgNameQueryInScene - image where find 
                          imgNameQuery
    
    '''
    img1 = cv2.imread(imgNameQuery, 0)  # queryImage
    img2 = cv2.imread(imgNameQueryInScene, 0)
    
    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    
    
    # Match descriptors.
    matches = bf.match(des1, des2)    
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # only 10 best results
    kpTmp = [kp2[i.trainIdx] for i in matches[:5]]

    # print(dir(i))

    # draw image and keypints kpTmp
    img3  = cv2.drawKeypoints(img2, kpTmp, color=(0, 255, 0), flags=0)
    
    # img3  = cv2.drawKeypoints(img1, kp1, color=(0, 255, 0), flags=0)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    return(img3)
