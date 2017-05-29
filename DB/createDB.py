'''
WARNINGS:
connect to database (even if not exists)
if you change table collumns (attributs of class)
you must delite existing tables

import classesTables as cT
p = cT.testTable(name="fedor",phone="noFone")  # insert data
cT.testTable.get(1)  # get data
'''


from sqlobject import sqlite, SQLObject
from sqlobject import StringCol, IntCol, BLOBCol


def load_paths():
    try:
        with open('DB/paths.txt') as f:
            paths = f.read()
        paths = eval(paths)
        print(paths)
        print("path to img.db: %s" % paths['img.db'])
        print("path to prepImg.db: %s" % paths['img.db'])
    except:
        print('ERROR: load_settings')
        paths = {'img.db':
                 '/media/valdecar/forData/data/projects/pythonDB/img.db',
                 'prepImg.db':
                 '/media/valdecar/forData/data/projects/pythonDB/prepImg.db'
             }

    return(paths)


paths = load_paths()

connection = sqlite.builder()(
    # '/media/valdecar/forData/data/projects/pythonDB/img.db'
    paths['img.db']
)


# create table
class imgDB(SQLObject):
    # every tables must has a pointer to database
    _connection = connection
    imgName = StringCol()
    imgOrign = StringCol()
    imgLabel = IntCol()


imgDB.createTable(ifNotExists=True)


# for storing features vectors space
connectionPrep = sqlite.builder()(
    # '/media/valdecar/forData/data/projects/pythonDB/prepImg.db'
    paths['prepImg.db']
)


class prepImgDB(SQLObject):
    _connection = connectionPrep
    imgPrep = BLOBCol()  # StringCol()
    imgLabel = IntCol()


prepImgDB.createTable(ifNotExists=True)


