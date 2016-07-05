import numpy as np
import cv2
#from matplotlib import pyplot as plt
from numpy import uint8,zeros,shape,ones,copy,array,float32
import DB.changeDB as cdb
from prepFuncts import createVectorEasy,createVectorWithHoG

#plt.imshow(im,cmap='Greys_r')
'''
#import imgBD
import DB.changeDB as cdb

#import images from database
imgs,labels=cdb.getAllImgsAndLabels(DB=cdb.imgDB)

#prepare imgs with HoG
imgPrep,labelsPrep=tm.prepareListOfImgs(imgs,labels)

#next 2 is not necessery steps:
  #save imgs to prepImgDB
  cdb.fillBdFromLists(imgPrep,labelsPrep,createNew=1)

  #load imgPrep,labelsPrep from prepImgDB
  imgPrep,labelsPrep=cdb.getAllImgsAndLabels1(DB=cdb.prepImgDB)

#train KNearest matcher
knn=train(imgPrep,labelsPrep)

#for some img
nearestLabels,distances=findNearest(img,knn)

#show result
imgs=showNearest(nearestLabels)
'''
class FeatureSpace():
    def __init__(self):
        self.imgs=None
        self.labels=None
        self.imgPrep=None
        self.labelsPrep=None
        self.knn=None
        self.nearestLabels=None
        self.distances=None
        self.prepOneImg=None
        self.nearestImgs=None
        self.nearestPrepImgs=None

    def getAllImgsAndLabels(self):
        self.imgs,self.labels=cdb.getAllImgsAndLabels(DB=cdb.imgDB)
        print("getAllImgs:done")
        print("len(imgs)=%s"%(len(self.imgs)))

    def prepareListOfImgs(self):
        if(self.imgs==None or self.labels==None):
            print("needed data empty")
            return(0)
        self.imgPrep,self.labelsPrep=prepareListOfImgs(self.imgs,self.labels)
        print("prepare:done")
        print("len(imgPrep)=%s"%(len(self.imgPrep)))
    
    def savePrepImgToDb(self):
        if(self.imgPrep==None or self.labelsPrep==None):
            print("needed data empty")
            return(0)
        cdb.fillBdFromLists(self.imgPrep,self.labelsPrep,createNew=1)
        print("save:done")
    
    def loadPrepImgAndLabelFromDb(self):
        self.imgPrep,self.labelsPrep=cdb.getAllImgsAndLabels(DB=cdb.prepImgDB)
        print("load prepImg: done")
        print("len(imgPrep)=%s"%(len(self.imgPrep)))

    def train(self):
        if(self.imgPrep==None or self.labelsPrep==None):
            print("needed data empty")
            return(0)
        self.knn=train(self.imgPrep,self.labelsPrep)
        print("train:done")

    def findNearest(self,img):
        if(img==None or self.knn==None):
            print("needed data empty")
            return(0)
        self.nearestLabels,self.distances,self.prepOneImg=findNearest(img,self.knn)
        print("findNearest:done")
        print("nearestLabels:")
        print(self.nearestLabels)

    def showNearest(self):
        if(self.nearestLabels==None):
            print("needed data empty")
            return(0)
        self.nearestImgs,self.nearestPrepImgs=showNearest(self.nearestLabels)
        print("showNearest:done")

def showNearest(nearestLabels,prepImgFunct=createVectorWithHoG):
    imgs=[]
    prepImgs=[]
    for label in nearestLabels:
        imgs.extend(cdb.getMultImgFromDbForLabel(int(label)))
        #prepImgs.extend(cdb.getMultImgFromDbForLabel(int(label),DB=cdb.prepImgDB))
    prepImgs=[prepImgFunct(img) for img in imgs]
    return((imgs,prepImgs))
    
def findNearest(img,knn):
    '''DESCRIPTION:
    find nearest class for vector img (it image  will be transform to vector in features space)
    INPUT:
    img - color image (from cv2.imread("",1))
    knn-trained KNearest object from train function
    OUTPUT:
    nearestLabels-
    distances
    '''
    img0=createVectorWithHoG(img)
    img1=img0.reshape(-1)
    print(img1)
    print(array([img1]).astype(float32))
    s=knn.find_nearest(array([img1]).astype(float32),3)
    nearestLabels=s[2].astype(int)
    distances=s[3]
    return(nearestLabels[0],distances[0],img0)

def train(listOfPrepImgs,listOfPrepLabels):
    '''DESCRIPTION:
    divide vecors in listOfPrepImgs at classes, shown in listOfPrepLabels
    INPUT:
    listOfPrepImgs  - from prepareListOfImgs
    listOfPrepLables- from ...
    '''
    knn=cv2.KNearest()
    knn.train(array(listOfPrepImgs).astype(float32),array(listOfPrepLabels).astype(float32))
    return(knn)


def prepareListOfImgs(listOfImgs,listOfLabels):
    '''DESCRIPTION:
    Transform listOfImgs into list of vectors for feature space
    (i.e. for knn).
    INPUT:
    listOfImgs -list of arrays representetion of images (each image from cv2.imread)
    listOfLabels-int label for each image in listOfImgs
    OUTPUT:
    listOfPrepImgs -vector for knn.train
    listOfPrepLabels- labels for knn.train
    '''
    #listOfImgs=[cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in listOfImgs]
    listOfPrepImgs=[]
    listOfPrepLabels=[]
    for i in range(len(listOfImgs)):
        imgPrep=createVectorWithHoG(listOfImgs[i])
        if imgPrep!=None:
            listOfPrepImgs.append(imgPrep.reshape(-1))
            listOfPrepLabels.append(listOfLabels[i])

    #listOfImgs=[createVectorWithHoG(img) for img in listOfImgs]
    return((listOfPrepImgs,listOfPrepLabels))
