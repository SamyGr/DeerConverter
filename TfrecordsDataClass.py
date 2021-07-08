# -*- coding: utf-8 -*-

import os
import io
import numpy as np
import requests
import tensorflow as tf

from object_detection.utils import dataset_util
from PIL import Image

from MediatorClass import Mediator, MediatorImages, MediatorCategories, MediatorBboxs

class TfrecordsReader: #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    def __init__(self):
        self.saveImgs = False
        self.outputImgsDir = './tfrec_imgs/'
        self.tfrecFname = None
        self.labelsFname = None
        
    def set_tfrec_file(self, tfrecFname):
        assert os.path.isfile(tfrecFname), 'Tfrecord file must exist.'
        self.tfrecFname = tfrecFname
        
    def set_labels_file(self, labelsFname):
        assert os.path.isfile(labelsFname), 'labels file must exist.'
        self.labelsFname = labelsFname
    
    def set_output_imgs(self, outputImgsDir):
        # create folder if not existing
        if not os.path.isdir(outputImgsDir):
            os.mkdir(outputImgsDir)
        self.outputImgsDir = os.path.abspath(outputImgsDir)
        
    def create_mediator_categ(self):
        mediatorCateg = MediatorCategories()
        
        # open labels file
        labels = open(self.labelsFname, 'r')
        while True:
            line = labels.readline().replace(" ", "").replace("\n", "")
            if line.startswith('item'):
                # get label number
                id_line = labels.readline().replace(" ", "").replace("\n", "")
                for word in id_line.split(":"):
                    if word.isdigit():
                        label_num = word
                # get label name
                label_line = labels.readline().replace(" ","").replace("\n", "")
                for word in label_line.split(":"):
                    if word.startswith("'") and word.endswith("'"):
                        label_name = word.replace("'", "")
                # add label to categories object
                mediatorCateg.append(ID=label_num, name=label_name)
            elif labels.read() == '': break
        labels.close()
        return(mediatorCateg)
    
    def create_mediator_imgs(self): #https://www.tensorflow.org/tutorials/load_data/tfrecord
        mediatorImgs = MediatorImages()
    
        # open .record file and extract informations
        dataset = tf.data.TFRecordDataset(self.tfrecFname)
        samples = dataset.map(self.extract_fn)
        
        # browse all images in .record file
        for image_features in samples:
            # decode image
            image_encoded = image_features["image/encoded"]
            dense = tf.sparse.to_dense(image_encoded)
            dense_scalar = tf.reshape(dense, [])
            image_raw = tf.io.decode_image(dense_scalar).numpy()
            
            # decode img informations
            height = image_features["image/height"].numpy()
            width = image_features["image/width"].numpy()
            filename = tf.sparse.to_dense(image_features['image/filename']).numpy()[0].decode('utf-8')
            sourceid = tf.sparse.to_dense(image_features["image/source_id"]).numpy()[0].decode('utf-8')
            format_ = tf.sparse.to_dense(image_features["image/format"]).numpy()[0].decode('utf-8')
            
            # get image depth
            depth = image_raw.shape[2]
            if depth == 1: image_raw = np.squeeze(np.array(image_raw), axis=2) # convert array if grayscale image
            
            # decode object bboxs
            xmins = tf.sparse.to_dense(image_features['image/object/bbox/xmin']).numpy()
            xmaxs = tf.sparse.to_dense(image_features['image/object/bbox/xmax']).numpy()
            ymins = tf.sparse.to_dense(image_features['image/object/bbox/ymin']).numpy()
            ymaxs = tf.sparse.to_dense(image_features['image/object/bbox/ymax']).numpy()
            classes = tf.sparse.to_dense(image_features['image/object/class/text']).numpy()
            labels_n = tf.sparse.to_dense(image_features['image/object/class/label']).numpy()
            
            mediatorBboxs = MediatorBboxs()
            for idx in range(len(xmins)):
                mediatorBboxs.append(ID=mediatorImgs.get_new_bbox_id(), labelID=labels_n[idx],
                                     x=xmins[idx]*width, y=ymins[idx]*height, 
                                     height=(ymaxs[idx]-ymins[idx])*height, width=(xmaxs[idx]+xmins[idx])*width)
            # save image if needed
            if self.saveImgs:
                filename = os.path.join(self.outputImgsDir, os.path.basename(filename))
                imgOut = Image.fromarray(image_raw)
                imgOut.save(filename)

                    
                
            mediatorImgs.append(path=filename, width=width, height=height, depth=depth, bboxs=mediatorBboxs, handlepath=True)
        return(mediatorImgs)
            
    def extract_fn(self, data_record): #https://stackoverflow.com/questions/54723912/tensorflow-extracting-image-and-label-from-tfrecords-file
        features = {# Extract features using the keys set during creation
                    "image/height":                 tf.io.FixedLenFeature([], tf.int64),
                    "image/width":                  tf.io.FixedLenFeature([], tf.int64),
                    "image/filename":               tf.io.VarLenFeature(tf.string),
                    "image/source_id":              tf.io.VarLenFeature(tf.string),
                    "image/encoded":                tf.io.VarLenFeature(tf.string),
                    "image/format":                 tf.io.VarLenFeature(tf.string),
                    "image/object/bbox/xmin":       tf.io.VarLenFeature(tf.float32),
                    "image/object/bbox/xmax":       tf.io.VarLenFeature(tf.float32),
                    "image/object/bbox/ymin":       tf.io.VarLenFeature(tf.float32),
                    "image/object/bbox/ymax":       tf.io.VarLenFeature(tf.float32),
                    "image/object/class/text":      tf.io.VarLenFeature(tf.string),
                    "image/object/class/label":     tf.io.VarLenFeature(tf.int64)}
        return tf.io.parse_single_example(data_record, features)
        
    def translate2mediator(self, tfrecFname, labelsFname, saveImgs=False, outputImgsDir='./tfrec_imgs/'):
        """
        Translate TFRecords annotation file to Mediator Class.
        
        input: tfrecFname: file path containing TFRecords images and annotations (most of the time: .record/.tfrecord/.records)
               labelsFname: file path containing labels informations (most of the time .pbtxt)
               saveImgs: enable images saving when reading TFRecords file. If this option is enabled, the images will be stored in outputImgsDir directory.
               outputImgsDir: effective when saveImgs is enabled. Directory where images will be stored.
               
        output: a Mediator class object containing annotations and classes informations
        
        for more informations about TFRecords annotation format: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        """
        # set variables
        self.set_tfrec_file(tfrecFname)
        self.set_labels_file(labelsFname)
        if saveImgs:
            self.saveImgs = saveImgs
            self.set_output_imgs(outputImgsDir)
        
        # create Mediator objects
        mediatorCateg = self.create_mediator_categ()
        print('[INFO] Loading TFRecords annotations...')
        mediatorImgs = self.create_mediator_imgs()
        return(Mediator(objImgs=mediatorImgs, objCateg=mediatorCateg))
    
    
class TfrecordsWriter: #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    def __init__(self):
        self.mediator = Mediator()
        self.dataFile = './tfrec_annot.record'
        self.labelsFname = './labels.pbtxt'
        self.enableDownload = False # in case of dataset read from coco, download is necessary to get images
        
    def set_enable_download(self, enableDownload):
        if enableDownload: print('[INFO] Download has been enable, conversion could take some minutes depending to your Internet connection.')
        self.enableDownload = enableDownload
    
    def set_mediator(self, mediator):
        assert type(mediator) is Mediator, 'mediator variables must be an Mediator object.'
        self.mediator = mediator
    
    def set_output_file(self, dataFile):
        self.dataFile = os.path.abspath(dataFile)
        
    def set_labels_file(self, labelsFname):
        self.labelsFname = os.path.abspath(labelsFname)
        
    def write_labelmap(self):
        labels = open(self.labelsFname, 'w')
        for classe in self.mediator.categList.list:
            labels.write("item  {\n")
            labels.write("  id: " + str(classe['id']) +"\n")
            labels.write("  name: '" + classe['name'] + "'\n") 
            labels.write("}\n")
        labels.close()
        
    def write_annot(self):
        writer = tf.io.TFRecordWriter(os.path.abspath(self.dataFile))
        
        # browse all images on our dataset
        for img in self.mediator.imgList.list:
            # check than the picture exist
            if os.path.isfile(img['path']):
                # import and encode image
                with tf.io.gfile.GFile(img['path'], 'rb') as fid: encoded_jpg = fid.read()
            # if datatset is read from COCO dataset, download images from urls
            elif self.enableDownload:
                if img['cocoURL']:
                    download_jpg = Image.open(requests.get(img['cocoURL'], stream=True).raw)
                elif img['flickrURL']:
                    download_jpg = Image.open(requests.get(img['flickrURL'], stream=True).raw)
                else:
                    raise(Exception('Download is enable but no URl is available for {} image. Check your URL assignment.'.format(img['fname'])))
                
                encoded_jpg = io.BytesIO()
                download_jpg.save(encoded_jpg, format='JPEG')
                encoded_jpg = encoded_jpg.getvalue()

            # raise error if file doesnt exist and download is disable
            else:
                 raise(Exception('Image at {} doesnt exist and download is not enable.\nIn case of reading from COCO dataset, enable download to construct .record file.'.format(img['path'])))
            
            # get image informations
            filename = img['fname'].encode('utf8')
            img_format = self.mediator.imgList.get_image_format(img['id']).encode('utf8')
            height = img['height']
            width = img['width']
            
            # initialize bboxs lists
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            labels = []
            
            # browse all bboxs of the current image
            for bbox in img['bboxs'].list:
                xmins.append(bbox['x'] / width)
                xmaxs.append((bbox['x']+bbox['width']) / width)
                ymins.append(bbox['y'] / height)
                ymaxs.append((bbox['y']+bbox['height']) / height)
                classes_text.append(self.mediator.categList.get_class_name(bbox['labelID'])[0].encode('utf8'))
                labels.append(bbox['labelID'])
            
            # create tf variable example
            tfExample = tf.train.Example(features=tf.train.Features(feature={
                                         'image/height': dataset_util.int64_feature(height),
                                         'image/width': dataset_util.int64_feature(width),
                                         'image/filename': dataset_util.bytes_feature(filename),
                                         'image/source_id': dataset_util.bytes_feature(filename),
                                         'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                                         'image/format': dataset_util.bytes_feature(img_format),
                                         'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                                         'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                                         'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                                         'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                                         'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                                         'image/object/class/label': dataset_util.int64_list_feature(labels)}))
            # add tfExample to .record file
            writer.write(tfExample.SerializeToString())
        writer.close()
    
    def write(self, mediator, outputAnnotFile='./tfrec_annot.record', outputLabelsFile='./labels.pbtxt', enableDownload=False):
        """
        Translate Mediator class object to TFRecord annotation file.
        
        input: mediator: Mediator object obtained by reading in another format
               outputAnnotFile: file path where .record annotation file will be stored
               outputLabelsFile: file path where .pbtxt labels file will be stored
               enableDownload: enable to download images in case of reading from COCO dataset (coco_url or flickr_url required)

        for more informations about TFRecords annotation format: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        """
        # set variables
        self.set_mediator(mediator)
        self.set_output_file(outputAnnotFile)
        self.set_labels_file(outputLabelsFile)
        self.set_enable_download(enableDownload)
        
        # write files
        self.write_labelmap()
        print("[INFO] Successfully created label map at {}".format(self.labelsFname))
        print("[INFO] Writing TFRecords annotation file...")
        self.write_annot()
        print("[INFO] Successfully created annotation file at {}".format(self.dataFile))
