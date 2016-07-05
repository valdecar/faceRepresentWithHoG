from createBD import imgDB
from createBD import prepImgDB
import pickle
#from numpy import array
'''
#for select with conditions (imgLabel == 5) 
#it return list 
imgs = imgDB.select(imDB.q.imgLabel==5)

'''
def getAllImgsAndLabels1(DB=prepImgDB):
    entrys=DB.select()
    imgs=[pickle.loads(entry.imgPrep) for entry in entrys]
    #imgLabel is int
    labels=[entry.imgLabel for entry in entrys]
    #for entry in entrys:
    #    imgs.append(pickle.loads(entry.imgOrign))
    return((imgs,labels))


def getAllImgsAndLabels(DB=imgDB):
    entrys=DB.select()
    if DB.__name__=='imgDB':
        imgs=[pickle.loads(entry.imgOrign) for entry in entrys]
    if DB.__name__=='prepImgDB':
        imgs=[pickle.loads(entry.imgPrep) for entry in entrys]
    
    #imgLabel is int
    labels=[entry.imgLabel for entry in entrys]
    #for entry in entrys:
    #    imgs.append(pickle.loads(entry.imgOrign))
    return((imgs,labels))

def fillBdFromLists(imgsVectors,imgsLabels,createNew=0):
    '''DESCRIPTION:
    fill prepImgDB from imgsVectors,imgsLabels lists
    INPUT:
    imgsVectors,imgsLabels from prepareListOfImgs function
    if createNew=1 database will be rewriten
    '''
    if createNew:
        #select=prepImgDB.select()
        #beter will be delete table
        prepImgDB.deleteMany(where=None)
        #prepImgDB.createTable(ifNotExists=False)

    for i in range(len(imgsVectors)):
        imgPrep = pickle.dumps(imgsVectors[i])
        imgLabel = imgsLabels[i]

        prepImgDB(imgPrep=imgPrep,imgLabel=imgLabel)
    print("done")

def fillBdFromFile(arrayFromFile):
    for i in range(len(arrayFromFile[0][0])):
        #print(arrayFromFile[0][0][i])
        imgName = arrayFromFile[0][0][i]
        imgOrign = pickle.dumps(arrayFromFile[0][2][i])
        imgLabel = arrayFromFile[0][1][i]

        imgDB(imgName=imgName,imgOrign=imgOrign,imgLabel=imgLabel)

def getOneImgFromDb():
    entry=imgDB.get(1)
    img=pickle.loads(entry.imgOrign)
    return(img)

def getMultImgFromDbForLabel(label,count=1,DB=imgDB):
    '''return  images with label
    count is count of returned images
    '''
    
    entrys=DB.select(DB.q.imgLabel==label)
    imgs=[]
    i=0
    for entry in entrys:
        if i>=count:
            break
        if DB.__name__=='imgDB':
            imgs.append(pickle.loads(entry.imgOrign))

        if DB.__name__=='prepImgDB':
            imgs.append(pickle.loads(entry.imgPrep))
        i=i+1
    return(imgs)
