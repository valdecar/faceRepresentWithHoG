description = '''With this tool one can create classes (labels) for images (manually)
from database and then transform that images to vectors of features space
(using HoG - histogram of oriented gradients method).
This space will be divided at created class (using knn classifier).
In the end one can use this features space for classify some particular image.
'''

descCreateClasses = '''create classes for each image in database
for ex: {men, women}, {old, young}, {phlegmatic, choleric}, 
{extrovert, introvert} and so on.'''
descClassifyImages = '''choice class for each image in database by hand
(use classes, created at previous step)'''

descLoadImages = '''load images from database for preparing
(time and memory expensive)'''

descPrepImgs = '''prepare loaded images using HoG
(transform to features space)
(time expensive)'''

descSavePrepImgs = '''save features space to database'''
descLoadSpace = '''load features space from database'''

descTrain = '''train knn use loaded space or space after preparing '''
descNearest = '''find nearest class for you image in features space
see result in command shell
(wait for a minute when loaded)'''
