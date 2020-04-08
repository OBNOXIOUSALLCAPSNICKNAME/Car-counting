
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from collections import Counter
from sklearn.decomposition import PCA, TruncatedSVD

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'default_graph'

#CWD_PATH = os.getcwd()
CWD_PATH = "C:/tensorflow1/models/research/object_detection/"

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
PATH_TO_VIDEO = "E:/examples/cars/test_video.mp4"

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap('C:/tensorflow1/models/research/object_detection/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')



import os
import cv2
import numpy as np

import sys

from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time




Colors = []
for i in range(20):
    Colors.append([229,43,80])
    Colors.append([255,191,0])
    Colors.append([153,102,204])
    Colors.append([251,206,177])
    Colors.append([127,255,212])

names = ['cream', 'cheese', 'fresh', 'trash']

def VisualizeBoxes(boxes, classes, image):
    if classes is None or len(classes) == 0:
        return image

    font = cv2.FONT_HERSHEY_DUPLEX
    width = np.size(image, 1)
    height = np.size(image, 0)

    for i in range(len(boxes)):

        x1 = int(boxes[i][0] )
        y1 = int(boxes[i][1] )

        x2 = int(boxes[i][2] )
        y2 = int(boxes[i][3] )

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        cv2.rectangle(image,pt1, pt2, Colors[int(classes[i]) - 1], 2)



def IoU(first, second):
    ymin_1, xmin_1, ymax_1, xmax_1 = first
    ymin_2, xmin_2, ymax_2, xmax_2 = second

    overlap_x1 = max(xmin_1, xmin_2)
    overlap_y1 = max(ymin_1, ymin_2)
    overlap_x2 = min(xmax_1, xmax_2)
    overlap_y2 = min(ymax_1, ymax_2)

    overlap_width = (overlap_x2 - overlap_x1)
    overlap_height = (overlap_y2 - overlap_y1)

    if (overlap_width < 0) or (overlap_height < 0):
        return 0.0

    first_area = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    second_area = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

    overlap_area = overlap_width * overlap_height
    union_area = first_area + second_area - overlap_area

    iou = overlap_area / union_area + 0.0000001
    return iou

def Area(bndbox):
    ymin, xmin, ymax, xmax = bndbox
    return (xmax - xmin) * (ymax - ymin)

treshold = 70

def CalcDist(point):
    #p1 = np.asarray([0, 520])
    #p2 = np.asarray([1470, 135])

    p1 = np.asarray([0, 345])
    p2 = np.asarray([1280, 25])

    p3 = np.asarray((point[0], point[1]))
    return int(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))

def near_to_ROI(bndbox, w, h):
    y1,x1,y2,x2 = bndbox
    dist = CalcDist((x2*w, y2*h))
    if dist < treshold:
        return True
    else:
        return False

def PreprocData(boxes, scores, classes, treshold, frame):

    w = np.size(frame, 1)
    h = np.size(frame, 0)

    prep_boxes = []
    prep_classes = []
    prep_scores = []

    for j in range(classes.size):
        if (classes[0][j] == 3.0) and scores[0][j] > treshold and Area(boxes[0][j]) < 0.025:
            prep_scores.append(scores[0][j])
            prep_classes.append(3.0)
            prep_boxes.append(boxes[0][j])

    for bndbox in prep_boxes:
        y1,x1,y2,x2 = bndbox

        bndbox[0] = x1 * w
        bndbox[1] = y1 * h
        bndbox[2] = x2 * w
        bndbox[3] = y2 * h

        bndbox[2] = int(bndbox[2] - bndbox[0])
        bndbox[3] = int(bndbox[3] - bndbox[1])





    length = len(prep_classes)
    i = 0
    j = 0
    del_index = 0

    while i < length:
        j = 0
        while j < length:
            if i != j and IoU(prep_boxes[i], prep_boxes[j]) > 0.4:

                length -= 1
                if i == length:
                    i -= 1

                del_index = j if prep_scores[i] > prep_scores[j] else i

                del prep_boxes[del_index]
                del prep_classes[del_index]
                del prep_scores[del_index]

                i = 0
                j = -1
            j += 1
        i += 1


    return prep_boxes, prep_classes, prep_scores

def draw_bboxes(frame, bbox_xyxy, identities):

    for i in range(len(bbox_xyxy)):

        x1 = int(bbox_xyxy[i][0])
        y1 = int(bbox_xyxy[i][1])
        x2 = int(bbox_xyxy[i][2])
        y2 = int(bbox_xyxy[i][3])

        coord_y  = y1-25 if y1 > 50 else y2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (133, 160, 22), 2)
        cv2.rectangle(frame, (x1, coord_y), (x2, coord_y+25), (133, 160, 22), cv2.FILLED)

        frame = cv2.putText(frame,
                            str(identities[i]),
                            (x1,coord_y+25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (2,2,16), 2, cv2.LINE_AA)


frame_count = 0

class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture(PATH_TO_VIDEO)

        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")

    def detect(self):
        while self.vdo.grab():
            global frame_count

            frame_count += 1

            _, frame = self.vdo.retrieve()

            #frame = cv2.resize(frame, (1280, 720))
            frame_expanded = np.expand_dims(frame, axis=0)

            start = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            end = time.time()
            print("\ndetect", round(end - start, 5))


            prep_boxes, prep_classes, prep_scores = PreprocData(
            boxes,
            scores,
            classes,
            0.5,
            frame)

            print(prep_boxes[0])
            print(prep_classes[0])
            print(prep_scores[0])
            print('============================')

            #VisualizeBoxes(prep_boxes, prep_classes, frame)

            #"""
            start = time.time()
            if prep_boxes is not None:
                outputs = self.deepsort.update(prep_boxes, prep_scores, frame)
            end = time.time()
            print("track", round(end - start, 5))

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                draw_bboxes(frame, bbox_xyxy, identities)
            #"""

            #image_path = 'E:/results/' + str(frame_count) + '.jpg'
            #cv2.imwrite(image_path, frame)

            cv2.imshow('Object detector', frame)
            if cv2.waitKey(1) == ord('q'):
                break



if __name__ == "__main__":
    import sys

    det = Detector()
    det.detect()
