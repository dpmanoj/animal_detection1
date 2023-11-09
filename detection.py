import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

        self.mot_tracker = Sort()
        self.model = tf.saved_model.load('saved_model/')

        PATH_TO_LABELS = 'saved_model/label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    def run_inference_for_single_frame(self, frame):
        image = np.asarray(frame)
        #input needs to be a tensor, so converting it
        input_tensor = tf.convert_to_tensor(image)
        #model expects input to be a batch, so adding an axis
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict



    
    def predict(self, frame):
        
        output_dict = self.run_inference_for_single_frame(frame)

        detection_classes = []
        detection_scores = []
        detection_boxes = []

        for i in range(len(output_dict['detection_classes'])):
            if output_dict['detection_scores'][i]>=0.5:
                detection_classes.append(output_dict['detection_classes'][i])
                detection_scores.append(output_dict['detection_scores'][i])
                detection_boxes.append(output_dict['detection_boxes'][i])
        
        return detection_boxes, detection_scores, detection_classes
    
    def track(self, boxes):
        
        boxes = np.array(boxes, dtype=float)
        
        track_bbs = self.mot_tracker.update(boxes)
        new_id = []
        id = track_bbs[:,4]
        
        for i in range(len(id)):
            new_id.append(int(id[i]))
        
        new_id = np.array(new_id)
        return new_id
    
    def visual(self, frame, boxes, classes, scores, new_id):

        boxes = np.array(boxes, dtype=float)
        new_id = np.array(new_id, dtype=int)
        if new_id.shape[0] == boxes.shape[0]:
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                boxes,
                classes,
                scores,
                self.category_index,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=4,
                track_ids=new_id
            )
            imshow("detections", frame)

        else:
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