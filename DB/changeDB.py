'''
HINTS:
# for select with conditions (imgLabel == 5) 
# it return list 
imgs = imgDB.select(imDB.q.imgLabel==5)

# try:
#    prepImgDB(imgPrep=str(im),imgLabel=1)
# except Exception as e: s=e
'''


from DB.createDB import imgDB
from DB.createDB import prepImgDB
import pickle


def get_all_imgs_and_labels_1(DB=prepImgDB):
    entrys = DB.select()
    
    # get images from bytes
    imgs = [pickle.loads(entry.imgPrep) for entry in entrys]
    
    # imgLabel is int
    labels = [entry.imgLabel for entry in entrys]
    
    return((imgs, labels))


def get_all_imgs_and_labels(DB=imgDB):
    entrys = DB.select()
    if DB.__name__ == 'imgDB':
        imgs = [pickle.loads(entry.imgOrign.encode('latin1'),
                             encoding='latin1')
                for entry in entrys]
    if DB.__name__ == 'prepImgDB':
        
        imgs = [pickle.loads(entry.imgPrep)
                for entry in entrys]
    
        # when using StringCol insted of BLOBCol
        # see also fillBdFromLists 
        # and prepImgDB definition
        # imgs = [pickle.loads(eval(entry.imgPrep))
        #         for entry in entrys]
    # imgLabel is int
    labels = [entry.imgLabel for entry in entrys]
        
    return((imgs, labels))


def fill_db_from_lists(imgsVectors, imgsLabels, createNew=0):
    '''DESCRIPTION:
    fill prepImgDB from imgsVectors,imgsLabels lists
    INPUT:
    imgsVectors,imgsLabels from prepareListOfImgs function
    if createNew=1 database will be rewriten
    '''
    if createNew:
        # select=prepImgDB.select()
        # beter will be delete table
        prepImgDB.deleteMany(where=None)
        # prepImgDB.createTable(ifNotExists=False)

    for i in range(len(imgsVectors)):
        imgPrep = pickle.dumps(imgsVectors[i])
        # print(imgPrep)
        imgLabel = imgsLabels[i]

        prepImgDB(imgPrep=imgPrep, imgLabel=imgLabel)
        
        # when using StringCol insted of BLOBCol
        # see also getAllImgsAndLabels 
        # and prepImgDB definition
        # prepImgDB(imgPrep=imgPrep, imgLabel=imgLabel)
    print("done")


def fill_db_from_file(arrayFromFile):
    for i in range(len(arrayFromFile[0][0])):
        # print(arrayFromFile[0][0][i])
        imgName = arrayFromFile[0][0][i]
        imgOrign = pickle.dumps(arrayFromFile[0][2][i])
        imgLabel = arrayFromFile[0][1][i]

        imgDB(imgName=imgName, imgOrign=imgOrign, imgLabel=imgLabel)


def get_one_img_from_db():
    entry = imgDB.get(1)
    img = pickle.loads(entry.imgOrign.encode('latin1'),
                       encoding='latin1')
    return(img)


def get_mult_img_from_db_for_label(label, count=1, DB=imgDB):
    '''return  images with label
    count is count of returned images
    '''
    
    entrys = DB.select(DB.q.imgLabel == label)
    imgs = []
    i = 0
    for entry in entrys:
        if i >= count:
            break
        if DB.__name__ == 'imgDB':
            imgs.append(pickle.loads(entry.imgOrign.encode('latin1'),
                                     encoding='latin1'))

        if DB.__name__ == 'prepImgDB':
            imgs.append(pickle.loads(entry.imgPrep.encode('latin1'),
                                     encoding='latin1'))
        i = i+1
    return(imgs)
