'''
#import imgBD
import DB.changeDB as cdb

#import images from database
imgs,labels = cdb.getAllImgsAndLabels(DB=cdb.imgDB)

#prepare imgs with HoG
imgPrep,labelsPrep = tm.prepareListOfImgs(imgs,labels)

#next 2 is not necessery steps:
  #save imgs to prepImgDB
  cdb.fillBdFromLists(imgPrep, labelsPrep, createNew=1)

  #load imgPrep,labelsPrep from prepImgDB
  imgPrep,labelsPrep = cdb.getAllImgsAndLabels1(DB=cdb.prepImgDB)

#train KNearest matcher
knn = train(imgPrep, labelsPrep)

#for some img
nearestLabels,distances = findNearest(img, knn)

#show result
imgs = showNearest(nearestLabels)
'''


import cv2
from numpy import array, float32

import DB.changeDB as cdb
from prepFuncts import create_vector_with_HoG


class FeatureSpace():
    def __init__(self):
        self.imgs = None
        self.labels = None
        self.imgPrep = None
        self.labelsPrep = None
        self.knn = None
        self.nearestLabels = None
        self.distances = None
        self.prepOneImg = None
        self.nearestImgs = None
        self.nearestPrepImgs = None

    def get_all_imgs_and_labels(self):

        self.imgs, self.labels = cdb.get_all_imgs_and_labels(DB=cdb.imgDB)
        print("getAllImgs:done")
        print("len(imgs)=%s" % (len(self.imgs)))

    def prepare_list_of_imgs(self):

        if(self.imgs is None or self.labels is None):
            print("needed data empty")
            return(0)

        self.imgPrep, self.labelsPrep = prepare_list_of_imgs(self.imgs,
                                                          self.labels)
        print("prepare:done")
        print("len(imgPrep)=%s" % (len(self.imgPrep)))
    
    def save_prep_img_to_db(self):
        if(self.imgPrep is None or self.labelsPrep is None):
            print("needed data empty")
            return(0)
        cdb.fill_db_from_lists(self.imgPrep, self.labelsPrep, createNew=1)
        print("save:done")
    
    def load_prep_img_and_label_from_db(self):
        self.imgPrep, self.labelsPrep = cdb.get_all_imgs_and_labels(
            DB=cdb.prepImgDB)
        print("load prepImg: done")
        print("len(imgPrep)=%s" % (len(self.imgPrep)))

    def train(self):
        if(self.imgPrep is None or self.labelsPrep is None):
            print("needed data empty")
            return(0)
        self.knn = train(self.imgPrep, self.labelsPrep)
        print("train:done")

    def find_nearest(self, img):
        if(img is None or self.knn is None):
            print("needed data empty")
            return(0)
        self.nearestLabels, self.distances, self.prepOneImg = find_nearest(img,
                                                                          self.knn)
        print("findNearest:done")
        print("nearestLabels:")
        print(self.nearestLabels)

    def show_nearest(self):
        if(self.nearestLabels is None):
            print("needed data empty")
            return(0)
        self.nearestImgs, self.nearestPrepImgs = show_nearest(self.nearestLabels)
        print("showNearest:done")


def show_nearest(nearestLabels, prepImgFunct=create_vector_with_HoG):
    imgs = []
    prepImgs = []
    for label in nearestLabels:
        imgs.extend(cdb.get_mult_img_from_db_for_label(int(label)))
        # prepImgs.extend(cdb.getMultImgFromDbForLabel(int(label),
        #                                              DB=cdb.prepImgDB))
    prepImgs = [prepImgFunct(img) for img in imgs]
    return((imgs, prepImgs))

    
def find_nearest(img, knn):
    '''
    DESCRIPTION:
    Find nearest class for vector img (it image
    will be transform to vector in features space)
    
    INPUT:
    img - color image (from cv2.imread("",1))
    knn-trained KNearest object from train function
    
    OUTPUT:
    nearestLabels-
    distances
    '''

    img0 = create_vector_with_HoG(img)
    img1 = img0.reshape(-1)
    print(img1)
    print(array([img1]).astype(float32))
    s = knn.findNearest(array([img1]).astype(float32), 3)
    nearestLabels = s[2].astype(int)
    distances = s[3]
    return(nearestLabels[0], distances[0], img0)


def train(listOfPrepImgs, listOfPrepLabels):
    '''
    DESCRIPTION:
    Divide vecors in listOfPrepImgs at classes,
    shown in listOfPrepLabels.

    INPUT:
    listOfPrepImgs  - from prepareListOfImgs
    listOfPrepLables- from ...
    '''
    knn = cv2.ml.KNearest_create()
    # print(type(listOfPrepLabels[0]))
    # print(type(listOfPrepImgs[0]))
    print(len(array(listOfPrepImgs).astype(float32)))
    print(len(array(listOfPrepLabels).astype(float32)))

    knn.train(array(listOfPrepImgs).astype(float32),
              cv2.ml.ROW_SAMPLE,
              array(listOfPrepLabels).astype(float32))
    return(knn)


def prepare_list_of_imgs(listOfImgs, listOfLabels):
    '''
    DESCRIPTION:
    Transform listOfImgs into list of vectors for feature space
    (i.e. for knn).
    
    INPUT:
    listOfImgs - list of arrays representetion of
                images (each image from cv2.imread)
    listOfLabels - int label for each image in listOfImgs
    
    OUTPUT:
    listOfPrepImgs -vector for knn.train
    listOfPrepLabels- labels for knn.train
    '''
    # listOfImgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #               for img in listOfImgs]

    listOfPrepImgs = []
    listOfPrepLabels = []
    # print("listOfLabels=")
    # print(listOfLabels)
    for i in range(len(listOfImgs)):
        imgPrep = create_vector_with_HoG(listOfImgs[i])
        if imgPrep is not None:
            listOfPrepImgs.append(imgPrep.reshape(-1))
            listOfPrepLabels.append(listOfLabels[i])
    # print("listOfPrepLabels=")
    # print(listOfPrepLabels)        
    # listOfImgs = [createVectorWithHoG(img) for img in listOfImgs]
    return((listOfPrepImgs, listOfPrepLabels))
