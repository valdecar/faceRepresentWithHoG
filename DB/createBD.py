from sqlobject import *

'''connect to database (even if not exists)
if you change table collimns (attributs of class) 
you must delite existing tables

import classesTables as cT
p = cT.testTable(name="fedor",phone="noFone")#insert data
cT.testTable.get(1)#get data
'''

connection = sqlite.builder()('/media/valdecar/forData/data/projects/pythonDB/img.db')

#create table
class imgDB(SQLObject):
    #every tables must has a pointer to database
    _connection = connection
    imgName = StringCol()
    imgOrign = StringCol()
    imgLabel = IntCol()

imgDB.createTable(ifNotExists=True)

#for storing features vectors space
connectionPrep = sqlite.builder()('/media/valdecar/forData/data/projects/pythonDB/prepImg.db')

class prepImgDB(SQLObject):
    _connection=connectionPrep
    imgPrep = StringCol()
    imgLabel = IntCol()

prepImgDB.createTable(ifNotExists=True)
