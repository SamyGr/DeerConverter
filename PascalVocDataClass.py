# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET

from MediatorClass import Mediator, MediatorImages, MediatorCategories, MediatorBboxs

class PascalVocReader:
    def __init__(self):
        self.allxml = []
        self.dataDir = None
    
    def set_data_dir(self, dataDir):
        assert os.path.isdir(dataDir), "Data path must be a directory"
        self.dataDir = os.path.abspath(dataDir)
        self.allxml = [f for f in os.listdir(self.dataDir) if f.endswith(".xml")]
    
    def create_mediator(self): #https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
        mediatorImgs = MediatorImages()
        mediatorCateg = MediatorCategories()
        
        for xmlFile in self.allxml:
            xmlFile = os.path.join(self.dataDir, xmlFile)
            tree = ET.parse(xmlFile)
            root = tree.getroot()
            
            height= int(root.find('size/height').text)
            width= int(root.find('size/width').text)
            depth= int(root.find('size/depth').text)
            
            try: path = root.find('path').text
            except: path = xmlFile.replace('.xml','.jpg')
            
            try: sourceName = root.find('source/database').text
            except: sourceName = 'Unknown'
            
            try: sourceAnnot = root.find('source/annotation').text
            except: sourceAnnot = 'Unknown'
            
            try: sourceImg = root.find('source/image').text
            except: sourceImg = 'Unknown'
            
            try: segmented = root.find('segmented').text
            except: segmented = 0
            
            mediatorBboxs = MediatorBboxs()
            for boxes in root.iter('object'):
                classe = boxes.find("name").text
                ymin = float(boxes.find("bndbox/ymin").text)
                xmin = float(boxes.find("bndbox/xmin").text)
                bboxW = float(boxes.find("bndbox/xmax").text) - xmin
                bboxH = float(boxes.find("bndbox/ymax").text) - ymin
                
                try: pose = boxes.find('pose').text
                except: pose = 'Unspecified'
                
                try: truncated = int(boxes.find('truncated').text)
                except: truncated = 0
                
                try: occluded = int(boxes.find('occluded').text)
                except: occluded = 0
                
                try: difficult = int(boxes.find('difficult').text)
                except: difficult = 0
                
                # handle class names
                mediatorCateg.append(name=classe)
                
                # handle bboxs of the current image
                mediatorBboxs.append(ID=mediatorImgs.get_new_bbox_id(), labelID=mediatorCateg.get_label_num(classe), 
                                     x=xmin, y=ymin, height=bboxH, width=bboxW,
                                     truncated=truncated, difficult=difficult, occluded=occluded, pose=pose)
            
            mediatorImgs.append(path=path, height=height, width=width, depth=depth, bboxs=mediatorBboxs,
                                sourceName=sourceName, sourceImg=sourceImg, sourceAnnot=sourceAnnot,
                                segmented=segmented, handlepath=True)
        return(Mediator(objImgs=mediatorImgs, objCateg=mediatorCateg))
    
    def translate2mediator(self, dataDir):
        """
        Translate PascalVOC annotations files to Mediator Class.
        
        input: dataDir: folder path containing images and PascalVOC annotations (images and annotations must have the same name)
               
        output: a Mediator class object containing annotations and classes informations
        
        for more informations about PascalVOC annotation format: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
        """
        # set variables
        self.set_data_dir(dataDir)
        
        # create Mediator objects
        print('[INFO] Loading PascalVOC annotations ...')
        pascalMed = self.create_mediator()
        return(pascalMed)
    
    
class PascalVocWriter:
    def __init__(self):
        self.mediator = Mediator()
        self.dataDir = './pascalvoc_annot/'
        
    def set_output_dir(self, outputAnnotDir):
        if not os.path.isdir(outputAnnotDir):
            os.mkdir(outputAnnotDir)
        self.dataDir = os.path.abspath(outputAnnotDir)
        
    def set_mediator(self, mediator):
        assert type(mediator) is Mediator, 'mediator variable must be a Mediator object.'
        self.mediator = mediator

    def write_annot(self):
        for imgObject in self.mediator.imgList.list:
            xmlFname = os.path.splitext(os.path.basename(imgObject['path']))[0] + '.xml'
            xmlFile = os.path.join(self.dataDir, xmlFname)
            
            # PascalVOC header
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "folder").text = imgObject['folder']
            ET.SubElement(annotation, "filename").text = imgObject['fname']
            ET.SubElement(annotation, "path").text = imgObject['path']
            
            source = ET.SubElement(annotation, "source")
            ET.SubElement(source, "database").text = imgObject['sourceName']
            if imgObject['sourceImg'] != 'Unknown':
                ET.SubElement(source, "annotation").text = imgObject['sourceImg']
            if imgObject['sourceAnnot'] != 'Unknown':
                ET.SubElement(source, "image").text = imgObject['sourceAnnot']
            
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(imgObject['width'])
            ET.SubElement(size, "height").text = str(imgObject['height'])
            ET.SubElement(size, "depth").text = str(imgObject['depth'])
            
            ET.SubElement(annotation, "segmented").text = str(imgObject['segmented'])
                
            # PascalVOC objects
            for bbox in imgObject['bboxs'].list:
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = str(self.mediator.categList.get_class_name(bbox['labelID'])[0])
                ET.SubElement(obj, "pose").text = bbox['pose']
                ET.SubElement(obj, "truncated").text = str(bbox['truncated'])
                ET.SubElement(obj, "difficult").text = str(bbox['difficult'])
                if bbox['occluded'] != 0:
                    ET.SubElement(obj, "occluded").text = str(bbox['occluded'])
                
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(bbox['x']))
                ET.SubElement(bndbox, "ymin").text = str(int(bbox['y']))
                ET.SubElement(bndbox, "xmax").text = str(int(bbox['x'] + bbox['width']))
                ET.SubElement(bndbox, "ymax").text = str(int(bbox['y'] + bbox['height']))
            
            tree = ET.ElementTree(annotation)
            self.tree_indent(annotation) #indent to smooth file viewing
            tree.write(xmlFile, 'UTF-8')
        
    
    def tree_indent(self, elem, level=0): #https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
        i = "\n" + level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.tree_indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i 

    def write(self, mediator, outputAnnotDir='./pascalvoc_annot/'):
        """
        Translate Mediator class object to PascalVOC annotations files.
        
        input: mediator: Mediator object obtained by reading in another format
               outputAnnotDir: folder path where annotations will be stored

        for more informations about PascalVOC annotation format: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
        """
        # set variables
        self.set_output_dir(outputAnnotDir)
        self.set_mediator(mediator)
        
        # write files
        print("[INFO] Writing PascalVOC annotations files ...")
        self.write_annot()
        print("[INFO] Successfully created annotations files at {}".format(self.dataDir))
