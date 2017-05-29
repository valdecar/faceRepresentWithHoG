'''
Use from faceRepresentWihtHoG folder
USAGE:
import _tests.tests as ts
t = ts.Test_faceDetect()
t.run()
'''

from numpy import array, mean, copy
from DB.changeDB import get_one_img_from_db
import faceObjectsSearchers as fos
import faceDetect as fd

# prepare classifiers
eye = fos.get_eye_pair_big()
eyeL = fos.get_eye_left()
eyeR = fos.get_eye_right()
nose = fos.get_nose()
mouth = fos.get_mouth()

# prepare images
img = get_one_img_from_db()
faceImg = copy(fos.find_face(img)[0])
eyesPairImg = fos.find_at_face_object(faceImg, eye)
eyeImg = fos.find_at_face_object(faceImg, eyeL)

testImages = [img, faceImg, eyesPairImg, eyeImg]


class Test_faceDetect():
    def __init__(self, imgs=testImages):
        
        self.img = imgs[0]
        self.faceImg = imgs[1]
        self.eyesPairImg = imgs[2]
        self.eyeImg = imgs[3]
        
        # name of tested methods
        actionsTypes = ['prep', 'face',
                        'elips', 'MMCC', 'find']
        self.actions = dict([[name, fd.__dict__[name]]
                             for name in fd.__dict__.keys()
                             if name.split('_')[0] in actionsTypes])

    def run(self):
        for actionName in self.actions.keys():
            if 'eye_' in actionName:
                try:
                    self.actions[actionName](self.eyeImg)
                    print('test '+actionName)
                    print('success')
                except:
                    print('test '+actionName)
                    print(' fail')
            elif 'find' in actionName:
                try:
                    self.actions[actionName](self.img)
                    print('test '+actionName)
                    print('success')
                except:
                    print('test '+actionName)
                    print(' fail')
            else:
                try:
                    self.actions[actionName](self.faceImg)
                    print('test '+actionName)
                    print('success')
                except:
                    print('test '+actionName)
                    print(' fail')
        print('done')


def test_base_math():
    
    aa = array([[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
                [[3, 4, 5], [4, 5, 6], [7, 8, 9]]])
    
    # GOAL:
    # eliminate image width and
    # hight dimentions
    aa.reshape((-1, 3))
    # RETURN:
    # array([[1, 2, 3],
    #   [2, 3, 4],
    #   [3, 4, 5],
    #   [2, 3, 4],
    #   [3, 4, 5],
    #   [4, 5, 6],
    #   [3, 4, 5],
    #   [4, 5, 6],
    #   [7, 8, 9]])

    # GOAL:
    # find color mean vector
    m = mean(aa.reshape((-1, 3)), axis=0)
    # RETURN:
    # array([ 3.22222222,  4.22222222,  5.22222222])

    # GOAL:
    # slime one of dimentions
    aa[:, 1:-1, :]
    # RETURN:
    # array([[[2, 3, 4]],
    #
    #   [[3, 4, 5]],
    #
    #   [[4, 5, 6]]])

    return(1)

