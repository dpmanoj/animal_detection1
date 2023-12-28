import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import requests
import json

import asyncio

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from cv2 import imshow
from sort import *

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class detection:

    def __init__(self):

        self.url = 'http://localhost:8501/v1/models/xada_gai:predict'
        # self.url = 'http://192.168.226.177:8501/v1/models/xada_gai:predict'

        self.mot_tracker = Sort()

        PATH_TO_LABELS = 'saved_model/1/label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    def run_inference_for_single_frame(self, frame):

        json_response = requests.post(self.url, data=frame)
        response = json.loads(json_response.text)

        output_dict = response['predictions'][0]
        output_dict['detection_classes'] = np.array(output_dict['detection_classes'], dtype=np.int64)

        return output_dict

    def predict(self, data):
        
        output_dict = self.run_inference_for_single_frame(data)

        detection_classes = []
        detection_scores = []
        detection_boxes = []

        for i in range(len(output_dict['detection_classes'])):
            if output_dict['detection_scores'][i]>=0.5:
                detection_classes.append(output_dict['detection_classes'][i])
                detection_scores.append(output_dict['detection_scores'][i])
                detection_boxes.append(output_dict['detection_boxes'][i])
        
        print(detection_classes)
        self.sendMessage(detection_classes)
        return detection_boxes, detection_scores, detection_classes
        
    
    def sendMessage(self, classes):
        for i in range(len(classes)):
            if classes[i] == 1:
                print("cow")
            elif classes[i] == 2:
                print("dog")
            elif classes[i] == 3:
                print("tiger")
            elif classes[i] == 4:
                print("elephant")

                
    
    def track(self, boxes):
        
        boxes = np.array(boxes, dtype=float)
        
        track_bbs = self.mot_tracker.update(boxes)
        new_id = []
        id = track_bbs[:,4]
        
        for i in range(len(id)):
            new_id.append(int(id[i]))
        
        new_id = np.array(new_id)
        return new_id
    
    def visual(self, frame, boxes, classes, scores, new_id=None):

        boxes = np.array(boxes, dtype=float)
        # new_id = np.array(new_id, dtype=int)
        # new_id = None
        # if new_id.shape[0] == boxes.shape[0]:
        #     vis_util.visualize_boxes_and_labels_on_image_array(
        #         frame,
        #         boxes,
        #         classes,
        #         scores,
        #         self.category_index,
        #         instance_masks=None,
        #         use_normalized_coordinates=True,
        #         line_thickness=4,
        #         # track_ids=new_id
        #     )
        #     imshow("detections", frame)

        # else:
        vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes,
                classes,
                scores,
                self.category_index,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=4,
        )
        imshow("detections", frame)