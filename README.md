# faceRepresentWithHoG
### description
```
With this tool one can create classes for images 
in database and then transform that images to vectors of features space
This space will be divided at created class (using knn classifier).
After that one can use this features space for classify some particular image
Features space created using Scale-Invariant Feature Transform
(http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0)
Default imgDB use one of these:http://www.face-rec.org/databases/
You can also create own image database in same format like imgDB (see DB/createBD.py)

```
### requirements
```
python2.7,opencv2,numpy,scipy,skimage,sqlobject
download imgDB.db and prepImg.db from
```
imgDB:
```
```
prepImg:

```
add path from imgDB in DB/createDB.py file in string
connection = sqlite.builder()('/media/valdecar/forData/data/projects/pythonDB/img.db')
also add path to prepImg database (even if it not exist) to same file in string:
connectionPrep = sqlite.builder()('/media/valdecar/forData/data/projects/pythonDB/prepImg.db')
```
### usage
```
python gui.py
```
