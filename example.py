# -*- coding: utf-8 -*-

from YoloDataClass import YoloReader, YoloWriter
from PascalVocDataClass import PascalVocReader, PascalVocWriter
from CocoDataClass import CocoReader, CocoWriter
from TfrecordsDataClass import TfrecordsReader, TfrecordsWriter


""" Uncomment the required reader. """
reader = YoloReader()
mediator = reader.translate2mediator(dataDir= './data/yolo_data/source', namesFname='./data/yolo_data/classes.names')

# reader = PascalVocReader()
# mediator = reader.translate2mediator(dataDir= './data/pascalvoc_data/source')

# reader = CocoReader()
# mediator = reader.translate2mediator(jsonFname='./data/coco_data/coco.json')

# reader = TfrecordsReader()
# mediator = reader.translate2mediator(tfrecFname='./data/tfrecord_data/tfrecord.records', labelsFname='./data/tfrecord_data/tfrecords.pbtxt',
#                                      saveImgs=True, outputImgsDir='./data/tfrecord_data/tfrec_imgs/')


""" Uncomment the required writer """
# writer = YoloWriter()
# writer.write(mediator=mediator, outputAnnotDir='./data/tfrecord_data/yolo_annot/', outputNamesFname='./data/tfrecord_data/yolo.names')

writer = PascalVocWriter()
writer.write(mediator=mediator, outputAnnotDir='./data/yolo_data/pascalvoc_annot/')

# writer = CocoWriter()
# writer.write(mediator=mediator, outputAnnotFile='./data/pascalvoc_data/coco.json')

# in case you're converting COCO annotations to TFRecord annotations, enableDownload is highly recommanded
# writer = TfrecordsWriter()
# writer.write(mediator=mediator, outputAnnotFile='./data/coco_data/tfrecord.records', outputLabelsFile='./data/coco_data/tfrecords.pbtxt',
#              enableDownload=True)