# -*- coding: utf-8 -*-

import os
from datetime import datetime
now = datetime.now()

# TODO: handle sub bboxs of pascalvoc

class Mediator:
    def __init__(self, *args, **kwargs):
        # MediatorImages variable
        self.imgList = kwargs.get('objImgs', MediatorImages)
        # MediatorCategories variable
        self.categList = kwargs.get('objCateg', MediatorCategories)
        
        # licenses (format COCO)
        self.licenses = kwargs.get('licenses', [{'id':1, 'url':'https://creativecommons.org/publicdomain/zero/1.0/', 'name':'Public Domain'}])
        
        # database informations
        self.infoYear = kwargs.get('infoYear', int(now.year))
        self.infoVersion = kwargs.get('infoVersion', "1.0")
        self.infoDes = kwargs.get('infoDes', "No information")
        self.infoCont = kwargs.get('infoCont', "No information")
        self.infoUrl = kwargs.get('infoUrl', "http://unknown.org")
        self.infoDateCreated = kwargs.get('infoDateCreated', str(now.date()))
        
    def set_obj_list(self, objList):
        assert type(objList) is MediatorImages, 'objList must be a MediatorImages object.'
        self.objects = objList
        self.set_imgs_IDs()
        
    def set_categ(self, objCateg):
        assert type(objCateg) is MediatorCategories, 'objCateg must be a MediatorCategories object.'
        self.categories = objCateg
        
    
class MediatorCategories:
    def __init__(self):
        self.list = []
        self.numClasses = 0
        self.numSuperC = 0
        
    def append(self, name, ID=None, supercategory='Unspecified'):
        if not ID: ID=self.numClasses+1
        
        if self.isnewClass(name, supercategory):
            if self.isnewSuperC(supercategory):
                self.numSuperC += 1
            self.list.append({'id':int(ID), 'name':name, 'supercategory':supercategory})
            self.numClasses += 1
    
    def isnewClass(self, name, supercategory):
        for classe in self.list:
            if classe['supercategory']==supercategory and classe['name']==name:
                return(False)
        return(True)
    
    def isnewSuperC(self, supercategory):
        for classe in self.list:
            if classe['supercategory']==supercategory:
                return(False)
        return(True)
    
    def get_num_classes(self):
        return(self.numClasses)
    
    def get_num_superc(self):
        return(self.numSuperC)
    
    def get_label_num(self, className, supercategory='Unspecified'):
        match = [classe for classe in self.list if classe['name']==className and classe['supercategory']==supercategory]
        if len(match)>1: raise(Exception('Multiple classes have the class name {} and superclass name {}. Check your data or add supercategory to differentiate same classes names.'.format(className, supercategory)))
        if len(match)==0: raise(Exception('No object has class name {} and superclass name {}.'.format(className, supercategory)))
        return(match[0]['id'])
    
    def get_class_name(self, labelNumber):
        match = [classe for classe in self.list if classe['id']==labelNumber]
        if len(match)>1: raise(Exception('Multiple classes have the ID {}. Check your ID assignment.'.format(labelNumber)))
        if len(match)==0: raise(Exception('No object has label ID {}.'.format(labelNumber)))
        return(match[0]['name'], match[0]['supercategory'])

    
class MediatorImages:
    def __init__(self):
        self.list = []
        self.numImgs = 0
        self.numBboxs = 0
        self.numSubboxs = 0
    
    def append(self, handlepath=True, *args, **kwargs):
        self.numImgs += 1
        
        ID = kwargs.get('ID', self.numImgs)
        path = kwargs.get('path', None)
        folder = kwargs.get('folder', None)
        fname = kwargs.get('fname', None)
        
        width = kwargs.get('width', 0)
        height = kwargs.get('height', 0)
        depth = kwargs.get('depth', 0)

        sourceName = kwargs.get('sourceName', 'Unknown')
        sourceImg = kwargs.get('sourceImg', 'Unknown')
        sourceAnnot = kwargs.get('sourceAnnot', 'Unknown')
        
        licenseID = kwargs.get('licenseID', 1)
        segmented = kwargs.get('segmented', 0)
        dateCaptured = kwargs.get('date_captured', None)
        cocoURL = kwargs.get('cocoURL', None)
        flickrURL = kwargs.get('flickrURL', None)

        bboxs = kwargs.get('bboxs', MediatorBboxs())
        
        # in case handlepath is enable
        if handlepath:
            if path and not fname and not folder:
                path = os.path.abspath(path)
                folder = os.path.basename(os.path.dirname(path))
                fname = os.path.basename(path)
            elif not path and fname and folder:
                path = os.path.join(folder, fname)
                path = os.path.abspath(path)
            elif fname and not path and not folder:
                path = os.path.abspath(fname)

        self.list.append({'id':ID, 'width':width, 'height':height, 'depth':depth,
                          'path':path, 'folder':folder, 'fname':fname, 'license':licenseID,
                          'sourceName':sourceName, 'sourceImg':sourceImg, 'sourceAnnot':sourceAnnot,
                          'segmented':segmented, 'date_captured':dateCaptured, 'flickrURL':flickrURL, 'cocoURL':cocoURL,
                          'bboxs':bboxs})
        
    def get_num_imgs(self):
        return(self.numImgs)
    
    def get_num_bboxs(self):
        return(self.numBboxs)
    
    def get_num_subboxs(self):
        return(self.numSubboxs)
    
    def get_object(self, ID):
        match = [obj for obj in self.list if obj['id']==ID]
        if len(match)>1: raise(Exception('Multiple objects have the ID {}. Check the ID assignment.'.format(ID)))
        if len(match)==0: raise(Exception('No object has ID {}.'.format(ID)))
        return(match[0])
    
    def get_arg_object(self, ID):
        match = [index for index, classe in enumerate(self.list) if classe['id']==ID]
        if len(match)>1: raise(Exception('Multiple objects have the ID {}. Check the ID assignment.'.format(ID)))
        if len(match)==0: raise(Exception('No object has ID {}.'.format(ID)))
        return(match[0])
    
    def get_image_format(self, ID):
        match = [obj for obj in self.list if obj['id']==ID]
        if len(match)>1: raise(Exception('Multiple objects have the ID {}. Check the ID assignment.'.format(ID)))
        if len(match)==0: raise(Exception('No object has ID {}.'.format(ID)))
        ext = os.path.basename(match[0]['path']).split(".")[1]
        return(ext)
    
    def get_new_bbox_id(self):
        self.numBboxs += 1
        return(self.numBboxs)
    
    def get_new_subbox_id(self):
        self.numSubboxs += 1
        return(self.numSubboxs)

class MediatorBboxs:
    def __init__(self):
        self.list = []
        self.numBboxs = 0

    def append(self, *args, **kwargs):
        ID = kwargs.get('ID', None)
        labelID = kwargs.get('labelID', 0)
        x = kwargs.get('x', None)
        y = kwargs.get('y', None)
        height = kwargs.get('height', None)
        width = kwargs.get('width', None)
        
        pose = kwargs.get('pose', 'Unspecified')
        truncated = kwargs.get('truncated', 0)
        occluded = kwargs.get('occluded', 0)
        difficult = kwargs.get('difficult', 0)
        iscrowd = kwargs.get('iscrowd', 0)
        
        self.list.append({'id':ID, 'labelID':labelID,
                          'x':x, 'y':y, 'width':width, 'height':height,
                          'pose':pose, 'truncated':truncated, 'occluded':occluded, 'difficult':difficult, 'iscrowd':iscrowd})
        self.numBboxs += 1
        
    def get_num_bboxs(self):
        return(self.numBboxs)
    
    def get_object(self, ID):
        match = [classe for classe in self.list if classe['id']==ID]
        if len(match)>1: raise(Exception('Multiple objects have the ID {}. Check the ID assignment.'.format(ID)))
        if len(match)==0: raise(Exception('No object has ID {}.'.format(ID)))
        return(match[0])
    
    
class MediatorSubBoxs:
    def __init__(self):
        self.list = []
        self.numSubBoxs = 0
        
    def append(self, *args, **kwargs):
        pass
        
        
        
        
        
    