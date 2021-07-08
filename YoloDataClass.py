# -*- coding: utf-8 -*-

import os
import csv
from PIL import Image

from MediatorClass import Mediator, MediatorImages, MediatorCategories, MediatorBboxs

class YoloReader:
    def __init__(self):
        self.alljpg = []
        self.alltxt = []
        self.mediatorCateg = MediatorCategories()
        self.mediatorImgs = MediatorImages()
        
        self.dataDir = None
        self.namesFname = None

    def set_data_dir(self, dataDir):
        assert os.path.isdir(dataDir), "Data path must be an existing directory."
        self.dataDir = os.path.abspath(dataDir)
        self.get_available_data()
        
    def set_names_fname(self, namesFname):
        assert os.path.isfile(namesFname), "Names file must exist."
        self.namesFname = os.path.abspath(namesFname)
        self.create_mediator_categ()
    
    def create_mediator_categ(self):
        index = 1
        self.mediatorCateg = MediatorCategories()
        for line in open(self.namesFname, 'r').readlines():
            self.mediatorCateg.append(ID=index, name=line.replace('\n',''))
            index += 1
    
    def get_available_data(self):
        self.alljpg = [f for f in os.listdir(self.dataDir) if f.endswith(".jpg")]
        self.alltxt = [f for f in os.listdir(self.dataDir) if f.endswith(".txt")]
        self.check_pairing()
    
    def check_pairing(self):
        # check than all jpg files have an associated txt file
        for idxJpg, jpgFile in enumerate(self.alljpg):
            find = False
            for idxTxt, txtFile in enumerate(self.alltxt):
                if os.path.splitext(jpgFile)[0] == os.path.splitext(txtFile)[0]:
                    find = True
                    break
            if not find:
                print("[INFO] {} has no annotation file. Check images and annotations are in the same folder.".format(jpgFile))
                self.alljpg.remove(jpgFile)
        
         # check than all txt files have an associated jpg file
        for idxTxt, txtFile in enumerate(self.alltxt):
            find = False
            for idxJpg, jpgFile in enumerate(self.alljpg):
                if os.path.splitext(jpgFile)[0] == os.path.splitext(txtFile)[0]:
                    find = True
                    break
            if not find:
                print("[INFO] {} has no image associated. Check images and annotations are in the same folder.".format(txtFile))
                self.alltxt.remove(txtFile)
        
    def create_mediator_imgs(self):
        self.mediatorImgs = MediatorImages()
        
        for imgFile in self.alljpg:
            imgFile = os.path.join(self.dataDir, imgFile)
            txtFile = os.path.splitext(imgFile)[0] + '.txt'
            # open annotations file with csv to use delimiters
            yoloAnnot = open(txtFile, 'r')
            readCSV = csv.reader(yoloAnnot, delimiter=' ')
            
            # open image to get its dimensions
            img = Image.open(imgFile)
            w, h = img.size
            depth = len(img.split())

            mediatorBboxs = MediatorBboxs()
            for row in readCSV:
                classID = int(row[0])
                bboxH = float(int(float(row[4]) * h))
                bboxW = float(int(float(row[3]) * w))
                xmin = int(float(row[1]) * w) - bboxW/2
                ymin = int(float(row[2]) * h) - bboxH/2
                
                mediatorBboxs.append(ID=self.mediatorImgs.get_new_bbox_id(), labelID=classID+1, # mediator categories ID start with 1
                                     x=xmin, y=ymin, height=bboxH, width=bboxW)
            self.mediatorImgs.append(path=imgFile, width=w, height=h, depth=depth, bboxs=mediatorBboxs, handlepath=True)
        yoloAnnot.close()
        
    def translate2mediator(self, dataDir, namesFname):
        """
        Translate Yolo files to Mediator Class.
        
        input: dataDir: folder path containing images and yolo annotations (images and annotations must have the same name)
               namesFname: file path containing classes names at Yolo format
               
        output: a Mediator Class object containing annotations and classes informations
        
    
        for more informations about Yolo annotation format: https://github.com/AlexeyAB/Yolo_mark/issues/60
        """
        # set variables
        self.set_data_dir(dataDir)
        self.set_names_fname(namesFname)
        
        # create classes dict 
        self.create_mediator_categ()
        
        # create mediator objects
        print('[INFO] Loading YOLO annotations ...')
        self.create_mediator_imgs()
        return(Mediator(objImgs=self.mediatorImgs, objCateg=self.mediatorCateg))
        
    
class YoloWriter:
    def __init__(self):
        self.mediator = Mediator()
        self.dataDir = './yolo_annot/'
        self.namesFname = './yolo.names'
        
    def set_output_dir(self, outputAnnotDir):
        if not os.path.isdir(outputAnnotDir):
            os.mkdir(outputAnnotDir)
        self.dataDir = os.path.abspath(outputAnnotDir)
        
    def set_output_namesfile(self, outputNamesFname):
        self.namesFname = os.path.abspath(outputNamesFname)
        
    def set_mediator(self, mediator):
        assert type(mediator) is Mediator, 'mediator variable must be a Mediator object.'
        self.mediator = mediator
        
    def write_names(self):
        names = open(self.namesFname, 'w')
        for index in range(self.mediator.categList.get_num_classes()):
            names.write("{}\n".format(self.mediator.categList.get_class_name(index+1)[0]))
        names.close()
        print("[INFO] Successfully created .names file at {}".format(self.namesFname))
    
    def write_annot(self):
        for imgObject in self.mediator.imgList.list:       
            txtFname = os.path.splitext(os.path.basename(imgObject['path']))[0] + '.txt'
            txtFile = os.path.join(self.dataDir, txtFname)
            yoloAnnot = open(txtFile, 'w')
            
            for bbox in imgObject['bboxs'].list:
                labelnum = int(bbox['labelID']-1)
                relW = float(bbox['width'] / imgObject['width'])
                relH = float(bbox['height'] / imgObject['height'])
                relX = float((bbox['x'] + bbox['width']/2) / imgObject['width'])
                relY = float((bbox['y'] + bbox['height']/2) / imgObject['height'])
            
                yoloAnnot.write("{} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(labelnum, relX, relY, relW, relH))
            yoloAnnot.close()
        
        
    def write(self, mediator, outputAnnotDir='./yolo_annot/', outputNamesFname='./yolo.names'):
        """
        Translate Mediator class object to Yolo annotations files.
        
        input: mediator: Mediator object obtained by reading in another format
               outputAnnotDir: folder path where annotations will be stored
               outputNamesFname: file path where classes names will be stored

        for more informations about Yolo annotation format: https://github.com/AlexeyAB/Yolo_mark/issues/60
        """
        # set variables
        self.set_mediator(mediator)
        self.set_output_dir(outputAnnotDir)
        self.set_output_namesfile(outputNamesFname)
        
        # write files
        self.write_names()
        print("[INFO] Successfully created label map at {}".format(self.namesFname))
        print("[INFO] Writing YOLO annotations files ...")
        self.write_annot()
        print("[INFO] Successfully created annotations files at {}".format(self.dataDir))
