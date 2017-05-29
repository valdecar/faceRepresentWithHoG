# faceRepresentWithHoG
### description
```
With this tool one can create classes (labels) for images (manually)
from database and then transform that images to vectors of features space
(using HoG - histogram of oriented gradients method).
This space will be divided at created class (using knn classifier).
In the end one can use this features space for classify some particular image.

Features space created using Scale-Invariant Feature Transform
(http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0)

Default imgDB use one of these:http://www.face-rec.org/databases/
You can also create own image database in same format like imgDB (see DB/createBD.py)

```
![alt tag](https://raw.githubusercontent.com/valdecar/faceRepresentWithHoG/master/screen_overview_HoG1.png)

### requirements
```
python3.4,opencv2,numpy,scipy,skimage,sqlobject
download imgDB.db and prepImg.db from
```
imgDB: https://drive.google.com/open?id=0B5urs7cbv2vJRDFxNGNOaWVRSFE
```
```
prepImg: https://drive.google.com/file/d/0B5urs7cbv2vJcVBBWnE5b0czV2s/view?usp=sharing

```
add path to "img.db" in "faceRepresentWithHoG/DB/paths.txt" file (in dictionary form) 

also add path to "prepImg.db" in "faceRepresentWithHoG/DB/paths.txt" file (in dictionary form)
```
### usage
```
python3 gui.py
```
