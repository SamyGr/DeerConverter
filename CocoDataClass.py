# -*- coding: utf-8 -*-

import os
import json

from MediatorClass import Mediator, MediatorImages, MediatorCategories, MediatorBboxs

class CocoReader:
    def __init__(self):
        self.jsonFname = None
        
    def set_json_file(self, jsonFile):
        assert os.path.isfile(jsonFile), 'Json file must exist.'
        self.jsonFname = os.path.abspath(jsonFile)

    def create_mediator(self):
        # open json file
        jsonOpen = open(self.jsonFname, 'r')
        jsonFile = json.load(jsonOpen)
        
        # get database informations
        infoYear = jsonFile['info']['year']
        infoVersion = jsonFile['info']['version']
        infoDes = jsonFile['info']['description']
        infoCont = jsonFile['info']['contributor']
        infoUrl = jsonFile['info']['url']
        infoDateCreated = jsonFile['info']['date_created']
        
        # get licences informations
        licenses = jsonFile['licenses']
        
        # get categories informations
        mediatorCateg = MediatorCategories()
        for categ in jsonFile['categories']:
            mediatorCateg.append(ID=categ['id'], name=categ['name'], supercategory=categ['supercategory'])
        
        # get images informations
        mediatorImgs = MediatorImages()
        print('[INFO] Loading COCO images informations ...')
        for imgInfo in jsonFile['images']:
            # those are optional
            try: dateCaptured = imgInfo['date_captured']
            except: dateCaptured = None
            
            try: flickrURL = imgInfo['flickr_url']
            except: flickrURL = None
            
            try: cocoURL = imgInfo['coco_url']
            except: cocoURL = None
            
            mediatorImgs.append(ID=imgInfo['id'], height=imgInfo['height'], width=imgInfo['width'], sourceName=infoDes,
                                fname=imgInfo['file_name'], path=imgInfo['file_name'], licenseID=imgInfo['license'],
                                date_captured=dateCaptured, flickrURL=flickrURL, cocoURL=cocoURL, bboxs=MediatorBboxs(), handlepath=False)
        # get annotations informations
        print('[INFO] Loading COCO annotations ...')
        for annotation in jsonFile['annotations']:
            arg = mediatorImgs.get_arg_object(annotation['image_id'])
            
            mediatorImgs.list[arg]['bboxs'].append(ID=annotation['id'], labelID=annotation['category_id'],
                                                   x=annotation['bbox'][0], y=annotation['bbox'][1],
                                                   height=annotation['bbox'][3], width=annotation['bbox'][2], iscrowd=annotation['iscrowd'])
        # create Mediator variable
        cocoMed = Mediator(infoYear=infoYear, infoVersion=infoVersion, infoDes=infoDes, infoCont=infoCont, 
                           infoUrl=infoUrl, infoDateCreated=infoDateCreated, licenses=licenses,
                           objImgs=mediatorImgs, objCateg=mediatorCateg)
        jsonOpen.close()
        return(cocoMed)
    
    def translate2mediator(self, jsonFname):
        """
        Translate COCO annotation file to Mediator Class.
        
        input: jsonFname: file path (.json) containing COCO annotations
               
        output: a Mediator class object containing annotations and classes informations
        
        for more informations about COCO annotation format: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
        """
        # set variables
        self.set_json_file(jsonFname)
        
        # create Mediator objects
        cocoMed = self.create_mediator()
        return(cocoMed)   
        
    
class CocoWriter:
    def __init__(self):
        self.mediator = Mediator()
        self.dataFile = './coco_annot.json'
        
    def set_mediator(self, mediator):
        self.mediator = mediator
        
    def set_output_file(self, outputAnnotFile):
        assert outputAnnotFile.endswith('.json'), 'output file must be a json file.'
        self.dataFile = os.path.abspath(outputAnnotFile)
        
    def write_annot(self):
        # set coco informations
        info = {'year': self.mediator.infoYear,
                'version': self.mediator.infoVersion,
                'description':self.mediator.infoDes,
                'contributor':self.mediator.infoCont,
                'url':self.mediator.infoUrl,
                'date_created': self.mediator.infoDateCreated}
        
        # set coco licenses
        licenses = self.mediator.licenses
        
        # set coco categories
        categories = []
        for categ in self.mediator.categList.list:
            categories.append({'id':categ['id'], 'name':categ['name'], 'supercategory':categ['supercategory']})
            
        # set coco images
        images = []
        annotations = []
        for imgObject in self.mediator.imgList.list:
            imgInfo = ({'id': imgObject['id'],
                        'width': imgObject['width'],
                        'height': imgObject['height'],
                        'file_name': imgObject['fname'],
                        'license': imgObject['license']})
            if imgObject['date_captured']: imgInfo['date_captured']=imgObject['date_captured']
            if imgObject['flickrURL']: imgInfo['flickr_url']=imgObject['flickrURL']
            if imgObject['cocoURL']: imgInfo['coco_url']=imgObject['cocoURL']
            images.append(imgInfo)
            
            # set coco annotations
            for annotObject in imgObject['bboxs'].list:
                annotations.append({'id': annotObject['id'],
                                    'image_id': imgObject['id'],
                                    'category_id': annotObject['labelID'],
                                    'bbox': [annotObject['x'], annotObject['y'], annotObject['width'], annotObject['height']],
                                    'iscrowd': annotObject['iscrowd'],
                                    'segmentation': [0,0,0],
                                    'area': 0.0})
        # create global variable for writing
        outJson = {'info':info,
                   'licenses':licenses,
                   'categories':categories,
                   'images':images,
                   'annotations':annotations}
        
        # write result on json file
        jsonFile = json.dumps(outJson)
        jsonOpen = open(self.dataFile, 'w')
        jsonOpen.write(jsonFile)
        jsonOpen.close()
        
        
    def write(self, mediator, outputAnnotFile='./coco_annot.json'):
        """
        Translate Mediator class object to COCO annotation file.
        
        input: mediator: Mediator object obtained by reading in another format
               outputAnnotFile: file path where .json annotation file will be stored

        for more informations about COCO annotation format: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
        """
        # set variables
        self.set_mediator(mediator)
        self.set_output_file(outputAnnotFile)
        
        # write files
        print("[INFO] Writing COCO annotation file...")
        self.write_annot()
        print("[INFO] Successfully created annotation file at {}".format(self.dataFile))
