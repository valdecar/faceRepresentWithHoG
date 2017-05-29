from tkinter import Tk, Toplevel
from tkinter import Frame, Button, Entry, Label
from tkinter import Listbox, StringVar, END
from tkinter import Radiobutton, EXTENDED, NORMAL
from tkinter.filedialog import askopenfilename

from PIL import Image
from PIL import ImageTk
import pickle
from cv2 import imread, resize

import trianglMathing as tm

from description import description, descCreateClasses
from description import descClassifyImages, descLoadImages
from description import descPrepImgs, descSavePrepImgs
from description import descTrain, descNearest, descLoadSpace

'''        
img =Image.open("claire.jpeg")
imgP=ImageTk.PhotoImage(img)
root=Tk()
w=Label(root,image=imgP)
w.image=imgP
w.pack()
root.mainloop()
'''


class Tool(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.grid(row=0, column=0)

        # description
        self.desc = description
        # print(description)
        self.labelDesc = Label(self, text=self.desc)
        self.labelDesc.grid(row=0, column=1)

        # create classes
        self.childResultList = ["flegmat", "chaler", "singv", "melanch"]
        # self.childRoot1 = 0
        self.buttonCreateClasses = Button(self,
                                          text="create classes",
                                          command=self.choice)
        self.buttonCreateClasses.grid(row=1, column=0)

        self.labelCreateClassesDesc = Label(self, text=descCreateClasses)
        self.labelCreateClassesDesc.grid(row=1, column=1)

        self.button = Button(self, text="show classes", command=self.show)
        self.button.grid(row=2, column=0)
        
        # ClassifyImages
        self.childData = 0
        self.buttonClassifyImage = Button(self, text="classify image", 
                                          command=self.classify)
        self.buttonClassifyImage.grid(row=3, column=0)

        self.labelClassifyImageDesc = Label(self, text=descClassifyImages)
        self.labelClassifyImageDesc.grid(row=3, column=1)

        self.button = Button(self,
                             text="show classify",
                             command=self.show_classify)
        # self.button.grid(row=3, column=0)
        
        self.labelDer = Label(self, text="*************")
        self.labelDer.grid(row=4, column=0)
        
        # work with features space
        self.fSpace = tm.FeatureSpace()
        
        # load images form database
        self.buttonLoadImgs = Button(self, 
                                     text="Load Images",
                                     command=self.fSpace.get_all_imgs_and_labels)
        self.buttonLoadImgs.grid(row=5, column=0)
        self.labelLoadImgsDesc = Label(self, text=descLoadImages)
        self.labelLoadImgsDesc.grid(row=5, column=1)
        
        # prepare images
        self.buttonPrepImgs = Button(self,
                                     text="Prepare Images",
                                     command=self.fSpace.prepare_list_of_imgs)
        self.buttonPrepImgs.grid(row=6, column=0)
        self.labelPrepImgsDesc = Label(self, text=descPrepImgs)
        self.labelPrepImgsDesc.grid(row=6, column=1)
        
        # save prepImgs to database
        self.buttonSavePrepImgs = Button(self,
                                         text="Save PrepImgs",
                                         command=self.fSpace.save_prep_img_to_db)
        self.buttonSavePrepImgs.grid(row=7, column=0)
        self.labelSavePrepImgsDesc = Label(self, text=descSavePrepImgs)
        self.labelSavePrepImgsDesc.grid(row=7, column=1)
                       
        # load prepared image from database
        self.buttonLoadSpace = Button(self,
                                      text="Load fSpace from \n database",
                                      command=self.fSpace.load_prep_img_and_label_from_db)
        self.buttonLoadSpace.grid(row=8, column=0)
        self.labelLoadSpaceDesc = Label(self, text=descLoadSpace)
        self.labelLoadSpaceDesc.grid(row=8, column=1)
        
        self.labelDer1 = Label(self, text="*************")
        self.labelDer1.grid(row=9, column=0)
        
        #train knn
        self.buttonTrain = Button(self,
                                  text="train",
                                  command=self.fSpace.train)
        self.buttonTrain.grid(row=10, column=0)
        self.labelTrainDesc = Label(self, text=descTrain)
        self.labelTrainDesc.grid(row=10, column=1)
        
        # find nearest
        self.buttonNearest = Button(self,
                                    text="find nearest",
                                    command=self.find_nearest)
        self.buttonNearest.grid(row=11, column=0)
        self.labelNearestDesc = Label(self, text=descNearest)
        self.labelNearestDesc.grid(row=11, column=1)
        
        # label for status
        self.varForStatus = StringVar()
        self.labelStatus = Label(self, textvariable=self.varForStatus)
        # self.labelStatus.grid(row=11, column=0)
        # self.varForStatus.set("Redy")
        
    def find_nearest(self):
       
        # open dialog for choicing file
        filename = askopenfilename(parent=self)
        img = imread(filename, 1)
        
        self.fSpace.find_nearest(img)
        self.fSpace.show_nearest()
        
        print("len(imgs)=%s" % (len(self.fSpace.nearestImgs)))
        print("len(prepImgs)=%s" % (len(self.fSpace.nearestPrepImgs)))
                
        root1 = Toplevel()
        child1 = ShowNearest(root1, self, self.fSpace, img)
        root1.wm_title("nearest neighbors")

    def show_classify(self):
        print(self.childData)

    def classify(self):
        
        self.childData = 0
        root = Toplevel()
        child1 = ClassifyImage(root, self)
        h = 100
        w = 100
        x = root.winfo_screenwidth()
        y = root.winfo_screenheight()
        # root.geometry('%dx%d+%d+%d' % (w, h,x/2-30,y/2-30))
        # child1.parentWindow=self

    def show(self):
        print(self.childResultList)

    def choice(self):
        self.childResultList = []
        root = Tk()
        child1 = CreateSets(root)
        child1.parentWindow = self
        h = 400
        w = 400
        x = root.winfo_screenwidth()
        y = root.winfo_screenheight()
        root.geometry('%dx%d+%d+%d' % (w, h, x/2-30, y/2-30))
    
        
class ShowNearest(Frame):

        def __init__(self, parent, parentWindow, fSpace, img):
            Frame.__init__(self, parent)
            self.parent = parent
            self.parentWindow = parentWindow
            self.grid(row=0, column=0)
            self.fSpace = fSpace
            self.img = img
                        
            img = Image.fromarray(self.img)
            print("new done")
            resizePrepImg = resize(self.fSpace.prepOneImg, (200, 200))
            prepOneImg = Image.fromarray(resizePrepImg)  # self.fSpace.prepOneImg
            
            imgP = ImageTk.PhotoImage(img)
            prepOneImgP = ImageTk.PhotoImage(prepOneImg)
            
            # show desc
            label = Label(self, text="Target image")
            label.grid(row=0, column=0)
            
            label1 = Label(self, text="Target feature vector")
            label1.grid(row=0, column=1)
            
            # scrolls not work
            # http://effbot.org/tkinterbook/canvas.htm
            # Tkinter.Canvas.create_image-method
            # http://effbot.org/tkinterbook/scrollbar.htm
            # scrollbar = Scrollbar(self)
            # scrollbar.grid(row=0, column=5)
            # canvas = Canvas(self, yscrollcommand=scrollbar.set)
            # canvas.create_image(image=imgP)
            # cancas.grid(row=0, column=4)
            
            # scrollbar.config(command=canvas.yview)
            
            # show imgP
            self.labelImg = Label(self, image=imgP)
            self.labelImg.image = imgP
            self.labelImg.grid(row=1, column=0)
            
            # show prepOneImgP
            self.labelImgP = Label(self, image=prepOneImgP)
            self.labelImgP.image = prepOneImgP
            self.labelImgP.grid(row=1, column=1)
            
            # show desc
            label = Label(self, text="nearest")
            label.grid(row=2, column=0)

            label = Label(self, text=str(self.fSpace.nearestLabels))
            label.grid(row=2, column=1)

            # show all Nearest with its prepares
            for i in range(len(self.fSpace.nearestImgs)):
                resizeImg = resize(self.fSpace.nearestImgs[i], (200, 200))
                img = Image.fromarray(resizeImg)  # self.fSpace.nearestImgs[i]
                resizePrepImg = resize(self.fSpace.nearestPrepImgs[i],
                                       (200, 200))
                prepOneImg = Image.fromarray(resizePrepImg)  # self.fSpace.nearestPrepImgs[i]
            
                imgP = ImageTk.PhotoImage(img)
                prepOneImgP = ImageTk.PhotoImage(prepOneImg)
            
                # show imgP
                labelImg = Label(self, image=imgP)
                labelImg.image = imgP
                labelImg.grid(row=i+3, column=0)
            
                # show prepOneImgP
                labelImgP = Label(self, image=prepOneImgP)
                labelImgP.image = prepOneImgP
                labelImgP.grid(row=i+3, column=1)
                        
            # delite old image from view
            # if self.label:
                # self.label.destroy()


class ClassifyImage:

    def __init__(self, master, parentWindow):
        
        frame = Frame(master)
        frame.pack()
        self.frame = frame

        # for destroy purpose
        self.master = master

        self.parentWindow = parentWindow
        
        # database with images
        self.imgDB = tm.cdb.imgDB
        # for database.get(item)
        self.item = 1
        
        self.labelImgPlace = Label(frame,
                                   text="Image from database \n (click next)")
        self.labelImgPlace.grid(row=0, column=1)

        # buttons for going trought imgDB
        self.buttonNext = Button(frame, text="Next",
                                 command=self.next)
        self.buttonNext.grid(row=2, column=0)

        self.buttonBack = Button(frame, text="Back",
                                 command=self.back)
        self.buttonBack.grid(row=3, column=0)

        # label for name of class
        self.varForLabel = StringVar()
        self.labelClassName = Label(frame, textvariable=self.varForLabel)
        self.labelClassName.grid(row=4, column=0)

        self.button = Button(frame, text="load image",
                             command=self.load_image)
        # self.button.grid(row=0, column=0)
        
        self.label = 0
        # self.label=Label(frame,image=imgP)
        # self.label.pack(side=RIGHT)
        
        self.things = self.parentWindow.childResultList  # ["cup","banana","pencil"]
        
        for i in range(len(self.things)):
            b = Radiobutton(self.frame, text=self.things[i], value=i)
            # write i-ths element from list to database as label 
            b.config(command=lambda iter=i: self.choice_class(iter))
            b.deselect()
            b.grid(row=i+1, column=2)
    
    def next(self):

        self.item = self.item + 1
        try:
            self.load_image_from_entry(self.item)
        except:
            print("end of imgDB")
            self.item = 1

    def back(self):
        self.item = self.item-1
        try:
            self.load_image_from_entry(self.item)
        except:
            print("begin of imgDB")
            self.item = 1

    def choice_class(self, labelIter):
        
        # change entry.imgLabel
        entry = self.imgDB.get(self.item)
        entry.imgLabel = labelIter
        self.varForLabel.set(entry.imgLabel)
        self.parentWindow.childData = self.things[labelIter]
        # self.master.destroy()
        
    def load_image_from_entry(self, item):
        
        entry = self.imgDB.get(item)
        
        # import label
        self.varForLabel.set(entry.imgLabel)
        
        # import image 
        img = pickle.loads(entry.imgOrign.encode('latin1'), encoding='latin1')
        img = Image.fromarray(img)
        imgP = ImageTk.PhotoImage(img)
        
        # delite old image from view
        if self.label:
            self.label.destroy()
        
        self.label = Label(self.frame, image=imgP)  # ,height=1500,width=1500
        self.label.image = imgP
        self.label.grid(row=0, column=1)
    
    def load_image(self):
        
        filename = askopenfilename(parent=self.frame)
        img = Image.open(filename)
        print(img.size)
        imgP = ImageTk.PhotoImage(img)
        print(imgP.height())
        
        if self.label:
            self.label.destroy()

        self.label = Label(self.frame, image=imgP)  # ,height=1500,width=1500
        self.label.image = imgP
        self.label.grid(row=0, column=1)


class CreateSets(Frame):

    def __init__(self, parent):
       
        # super(createSets,self).__init__(parent)
        Frame.__init__(self, parent)
        self.parent = parent
        self.grid(row=0, column=0)

        self.parentWindow = 0

        self.listBox = Listbox(self, selectmode=EXTENDED)
        self.listBox.grid(row=1, column=1)
        for item in ["one", "two", "three", "four"]:
            self.listBox.insert(END, item)
        
        self.buttonDel = Button(self,
                                text="delite selected class",
                                command=self.del_selected)  # lambda ld=self.listBox:ld.delete(ANCHOR))
        self.buttonDel.grid(row=0, column=0)
            
        self.entry = Entry(self, state=NORMAL)
        # self.entry.focus_set()
        self.entry.insert(0, "default")
        self.entry.grid(row=1, column=0)
        
        self.buttonInsert = Button(self, text="add new class",
                                   command=self.add)
        self.buttonInsert.grid(row=0, column=1)
        
        self.buttonDone = Button(self, text="done", command=self.done)
        self.buttonDone.grid(row=2, column=0)

    def done(self):
        self.parentWindow.childResultList = self.listBox.get(0, END)
        # print(self.listBox.get(0, END))
        self.parent.destroy()
        
    def add(self):
        text = self.entry.get()
        self.listBox.insert(END, text)

    def del_selected(self):
        lb = self.listBox
        items = map(int, lb.curselection())
        
        for item in items:
            lb.delete(item)


if __name__ == "__main__":
    root = Tk()
    app = Tool(root)
    h = 500  # app.winfo_height()
    w = 700  # app.winfo_width()
    x = root.winfo_screenwidth()
    y = root.winfo_screenheight()
    print('%dx%d+%d+%d' % (w, h, x/2, y/2))
    root.geometry('%dx%d+%d+%d' % (w, h, x/2, y/2))
    root.wm_title("images classifier")
    root.mainloop()

        
def main():

    root = Tk()
    h = root.winfo_height()
    w = root.winfo_width()
    x = root.winfo_screenwidth()
    y = root.winfo_screenheight()
    root.geometry('%dx%d+%d+%d' % (w, h, x/2, y/2))
    app = Tool(root)
    # root.mainloop()
    return(root)
