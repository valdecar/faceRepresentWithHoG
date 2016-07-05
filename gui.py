from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image
from PIL import ImageTk
import trianglMathing as tm
import pickle
from cv2 import imread,resize
'''        
img =Image.open("claire.jpeg")
imgP=ImageTk.PhotoImage(img)
root=Tk()
w=Label(root,image=imgP)
w.image=imgP
w.pack()
root.mainloop()
'''
description='''With this tool one can create classes for images 
in database and then transform that images to vectors of features space
This space will be divided at created class (using knn classifier).
In the end one can use this features space for classify some particular image '''
descCreateClasses='''create classes for each image in database'''
descClassifyImages='''choice class for each image in database by hand 
(use classes, created at previous step)'''
descLoadImages='''load images from database for preparing'''
descPrepImgs='''prepare loaded image (transform to features space)'''
descSavePrepImgs='''save features space to database'''
descLoadSpace='''load features space from database'''
descTrain='''train knn use loaded space or space after preparing '''
descNearest='''find nearest class for you image in features space'''

class Tool(Frame):
    def __init__(self,parent):
        Frame.__init__(self,parent)
        self.parent=parent
        self.grid(row=0,column=0)

        #description
        self.desc=description
        #print(description)
        self.labelDesc=Label(self,text=self.desc)
        self.labelDesc.grid(row=0,column=1)

        #create classes
        self.childResultList=["flegmat","chaler","singv","melanch"]
        #self.childRoot1=0
        self.buttonCreateClasses=Button(self,text="create classes",command=self.choice)
        self.buttonCreateClasses.grid(row=1,column=0)
        self.labelCreateClassesDesc=Label(self,text=descCreateClasses)
        self.labelCreateClassesDesc.grid(row=1,column=1)
        self.button=Button(self,text="show classes",command=self.show)
        self.button.grid(row=2,column=0)
        
        #ClassifyImages
        self.childData=0
        self.buttonClassifyImage=Button(self,text="classify image",command=self.classify)
        self.buttonClassifyImage.grid(row=3,column=0)
        self.labelClassifyImageDesc=Label(self,text=descClassifyImages)
        self.labelClassifyImageDesc.grid(row=3,column=1)
        self.button=Button(self,text="show classify",command=self.showClassify)
        #self.button.grid(row=3,column=0)
        
        self.labelDer=Label(self,text="*************")
        self.labelDer.grid(row=4,column=0)
        
        #work with features space
        self.fSpace=tm.FeatureSpace()
        
        #load images form database
        self.buttonLoadImgs=Button(self,text="Load Images",command=self.fSpace.getAllImgsAndLabels)
        self.buttonLoadImgs.grid(row=5,column=0)
        self.labelLoadImgsDesc=Label(self,text=descLoadImages)
        self.labelLoadImgsDesc.grid(row=5,column=1)
        
        #prepare images
        self.buttonPrepImgs=Button(self,text="Prepare Images",command=self.fSpace.prepareListOfImgs)
        self.buttonPrepImgs.grid(row=6,column=0)
        self.labelPrepImgsDesc=Label(self,text=descPrepImgs)
        self.labelPrepImgsDesc.grid(row=6,column=1)
        
        #save prepImgs to database
        self.buttonSavePrepImgs=Button(self,text="Save PrepImgs",command=self.fSpace.savePrepImgToDb)
        self.buttonSavePrepImgs.grid(row=7,column=0)
        self.labelSavePrepImgsDesc=Label(self,text=descSavePrepImgs)
        self.labelSavePrepImgsDesc.grid(row=7,column=1)
               
        
        #load prepared image from database
        self.buttonLoadSpace=Button(self,text="Load fSpace from \n database",command=self.fSpace.loadPrepImgAndLabelFromDb)
        self.buttonLoadSpace.grid(row=8,column=0)
        self.labelLoadSpaceDesc=Label(self,text=descLoadSpace)
        self.labelLoadSpaceDesc.grid(row=8,column=1)
        
        #train knn
        self.buttonTrain=Button(self,text="train",command=self.fSpace.train)
        self.buttonTrain.grid(row=9,column=0)
        self.labelTrainDesc=Label(self,text=descTrain)
        self.labelTrainDesc.grid(row=9,column=1)
        
        #find nearest
        self.buttonNearest=Button(self,text="find nearest",command=self.findNearest)
        self.buttonNearest.grid(row=10,column=0)
        self.labelNearestDesc=Label(self,text=descNearest)
        self.labelNearestDesc.grid(row=10,column=1)
        
        #label for status
        self.varForStatus=StringVar()
        self.labelStatus=Label(self,textvariable=self.varForStatus)
        #self.labelStatus.grid(row=11,column=0)
        #self.varForStatus.set("Redy")
        
    def findNearest(self):
        #img=imread("/media/valdecar/forData/data/projectsNew/python/faceRepresentWithHoG/claire.jpeg")#"claire.jpeg",1)  
        #open dialog for choicing file
        filename = askopenfilename(parent=self)
        img=imread(filename,1)
        
        self.fSpace.findNearest(img)
        self.fSpace.showNearest()
        
        print("len(imgs)=%s"%(len(self.fSpace.nearestImgs)))
        print("len(prepImgs)=%s"%(len(self.fSpace.nearestPrepImgs)))
        
        
        root1=Toplevel()
        child1=ShowNearest(root1,self.fSpace,img)
        
    def showClassify(self):
        print(self.childData)

    def classify(self):
        self.childData=0
        root=Toplevel()
        child1=ClassifyImage(root,self)
        h=100
        w=100
        x=root.winfo_screenwidth()
        y=root.winfo_screenheight()
        #root.geometry('%dx%d+%d+%d' % (w, h,x/2-30,y/2-30))
        #child1.parentWindow=self

    def show(self):
        print(self.childResultList)

    def choice(self):
        self.childResultList=[]
        root=Tk()
        child1=CreateSets(root)
        child1.parentWindow=self
        h=400
        w=400
        x=root.winfo_screenwidth()
        y=root.winfo_screenheight()
        root.geometry('%dx%d+%d+%d' % (w, h,x/2-30,y/2-30))
    
        
class ShowNearest(Frame):
        def __init__(self,parent,fSpace,img):
            Frame.__init__(self,parent)
            self.parent=parent
            self.grid(row=0,column=0)
            self.fSpace=fSpace
            self.img=img
            
            
            
            img=Image.fromarray(self.img)
            resizePrepImg=resize(self.fSpace.prepOneImg,(200,200))
            prepOneImg=Image.fromarray(resizePrepImg)#self.fSpace.prepOneImg
            
            imgP=ImageTk.PhotoImage(img)
            prepOneImgP=ImageTk.PhotoImage(prepOneImg)
            
            #show desc
            label=Label(self,text="Target image")
            label.grid(row=0,column=0)
            
            #scrolls not work
            #http://effbot.org/tkinterbook/canvas.htm#Tkinter.Canvas.create_image-method
            #http://effbot.org/tkinterbook/scrollbar.htm
            #scrollbar = Scrollbar(self)
            #scrollbar.grid(row=0,column=5)
            #canvas = Canvas(self,yscrollcommand=scrollbar.set)
            #canvas.create_image(image=imgP)
            #cancas.grid(row=0,column=4)
            
            #scrollbar.config(command=canvas.yview)
            

            #show imgP
            self.labelImg=Label(self,image=imgP)
            self.labelImg.image=imgP
            self.labelImg.grid(row=1,column=0)
            
            #show prepOneImgP
            self.labelImgP=Label(self,image=prepOneImgP)
            self.labelImgP.image=prepOneImgP
            self.labelImgP.grid(row=1,column=1)
            
            #show desc
            label=Label(self,text="nearest")
            label.grid(row=2,column=0)

            #show all Nearest with its prepares
            for i in range(len(self.fSpace.nearestImgs)):
                resizeImg=resize(self.fSpace.nearestImgs[i],(200,200))
                img=Image.fromarray(resizeImg)#self.fSpace.nearestImgs[i]
                resizePrepImg=resize(self.fSpace.nearestPrepImgs[i],(200,200))
                prepOneImg=Image.fromarray(resizePrepImg)#self.fSpace.nearestPrepImgs[i]
            
                imgP=ImageTk.PhotoImage(img)
                prepOneImgP=ImageTk.PhotoImage(prepOneImg)
            
                #show imgP
                labelImg=Label(self,image=imgP)
                labelImg.image=imgP
                labelImg.grid(row=i+3,column=0)
            
                #show prepOneImgP
                labelImgP=Label(self,image=prepOneImgP)
                labelImgP.image=prepOneImgP
                labelImgP.grid(row=i+3,column=1)
            
            

            
            #delite old image from view
            #if self.label:
                #self.label.destroy()

class ClassifyImage:
    def __init__(self,master,parentWindow):
        
        frame=Frame(master)
        frame.pack()
        self.frame=frame

        #for destroy purpose
        self.master=master

        self.parentWindow=parentWindow
        
        #database with images
        self.imgDB=tm.cdb.imgDB
        #for database.get(item)
        self.item=1
        
        self.labelImgPlace=Label(frame,text="Image from database \n (click next)")
        self.labelImgPlace.grid(row=0,column=1)

        #buttons for going trought imgDB
        self.buttonNext=Button(frame,text="Next",command=self.Next)
        self.buttonNext.grid(row=2,column=0)
        self.buttonBack=Button(frame,text="Back",command=self.Back)
        self.buttonBack.grid(row=3,column=0)
        #label for name of class
        self.varForLabel=StringVar()
        self.labelClassName=Label(frame,textvariable=self.varForLabel)
        self.labelClassName.grid(row=4,column=0)

        self.button=Button(frame,text="load image",command=self.loadImage)
        #self.button.grid(row=0,column=0)
        
        self.label=0
        #self.label=Label(frame,image=imgP)
        #self.label.pack(side=RIGHT)
        
        self.things=self.parentWindow.childResultList#["cup","banana","pencil"]
        
        for i in range(len(self.things)):
            b=Radiobutton(self.frame,text=self.things[i],value=i)
            #write i-ths element from list to database as label 
            b.config(command=lambda iter=i:self.choiceClass(iter))
            b.deselect()
            b.grid(row=i+1,column=2)
    
    def Next(self):
        self.item=self.item+1
        try:
            self.loadImageFromEntry(self.item)
        except:
            print("end of imgDB")
            self.item=1

    def Back(self):
        self.item=self.item-1
        try:
            self.loadImageFromEntry(self.item)
        except:
            print("begin of imgDB")
            self.item=1

    def choiceClass(self,labelIter):
        #pass
        #change entry.imgLabel
        entry=self.imgDB.get(self.item)
        entry.imgLabel=labelIter
        self.varForLabel.set(entry.imgLabel)
        self.parentWindow.childData=self.things[labelIter]
        ##self.master.destroy()
        
    def loadImageFromEntry(self,item):
        entry=self.imgDB.get(item)
        #import label
        self.varForLabel.set(entry.imgLabel)
        #import image 
        img=pickle.loads(entry.imgOrign)
        img=Image.fromarray(img)
        imgP=ImageTk.PhotoImage(img)
        
        #delite old image from view
        if self.label:
            self.label.destroy()
        
        self.label=Label(self.frame,image=imgP)#,height=1500,width=1500
        self.label.image=imgP
        self.label.grid(row=0,column=1)
    
    def loadImage(self):
        
        filename = askopenfilename(parent=self.frame)
        #f = open(filename)
        #f.read()
        img=Image.open(filename)
        #img =Image.open("/media/valdecar/forData/data/projectsNew/python/faceRepresentWithHoG/claire.jpeg")#"claire.jpeg"
        print(img.size)
        imgP=ImageTk.PhotoImage(img)
        print(imgP.height())
        
        if self.label:
            self.label.destroy()

        self.label=Label(self.frame,image=imgP)#,height=1500,width=1500
        self.label.image=imgP
        self.label.grid(row=0,column=1)

class CreateSets(Frame):
    def __init__(self,parent):
        #super(createSets,self).__init__(parent)
        Frame.__init__(self,parent)
        self.parent=parent
        self.grid(row=0,column=0)

        self.parentWindow=0

        self.listBox=Listbox(self,selectmode=EXTENDED)
        self.listBox.grid(row=1,column=1)
        for item in ["one", "two", "three", "four"]:
            self.listBox.insert(END, item)
        
        self.buttonDel=Button(self,text="delite selected class",command=self.delSelected)#lambda ld=self.listBox:ld.delete(ANCHOR))
        
        self.buttonDel.grid(row=0,column=0)
    
        
        self.entry=Entry(self,state=NORMAL)
        #self.entry.focus_set()
        self.entry.insert(0,"default")
        self.entry.grid(row=1,column=0)
        
        self.buttonInsert=Button(self,text="add new class",command=self.add)
        self.buttonInsert.grid(row=0,column=1)
        
        self.buttonDone=Button(self,text="done",command=self.done)
        self.buttonDone.grid(row=2,column=0)

    def done(self):
        #pass
        self.parentWindow.childResultList=self.listBox.get(0,END)
        #print(self.listBox.get(0,END))
        self.parent.destroy()
        
    def add(self):
        text=self.entry.get()
        self.listBox.insert(END,text)

    def delSelected(self):
        lb=self.listBox
        items=map(int,lb.curselection())
        #print(items)
        for item in items:
            lb.delete(item)


if __name__=="__main__":
    root=Tk()
    app=Tool(root)
    h=500#app.winfo_height()
    w=700#app.winfo_width()
    x=root.winfo_screenwidth()
    y=root.winfo_screenheight()
    print('%dx%d+%d+%d' % (w, h,x/2,y/2))
    root.geometry('%dx%d+%d+%d' % (w, h,x/2,y/2))
    root.wm_title("images classifier")
    root.mainloop()

        

def main():
    root=Tk()
    h=root.winfo_height()
    w=root.winfo_width()
    x=root.winfo_screenwidth()
    y=winfo_screenheight()
    root.geometry('%dx%d+%d+%d' % (w, h,x/2,y/2))
    app=Tool(root)
    #root.mainloop()
    return(root)
