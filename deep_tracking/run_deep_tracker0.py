import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.decomposition import PCA, TruncatedSVD

import cv2
import numpy as np

from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time

#from yolo.models import *
#from yolo.utils import *

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


PATH_TO_VIDEO = './tests/video/2.mp4'
config_path='./cfg/yolov3.cfg'
weights_path='./weights/best.pt'
class_path='./data/1 cls/classes.names'

img_size=416
conf_thres=0.6
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image

    img = Image.fromarray(img)

    #width = np.size(img, 1)
    #height = np.size(img, 0)

    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def preproc_data(detections, img):

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    prep_boxes = []
    prep_classes = []
    prep_scores = []

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            y2 = ((y2 - pad_y // 2) / unpad_h) * img.shape[0]
            x2 = ((x2 - pad_x // 2) / unpad_w) * img.shape[1]

            x2 = x2 - x1
            y2 = y2 - y1

            prep_boxes.append([x1, y1, x2, y2])
            prep_classes.append(int(cls_pred))
            prep_scores.append(float(conf))

    return prep_boxes, prep_classes, prep_scores

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

            #frame = cv2.resize(frame, (1920, 1080))
            #frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

            detections = detect_image(frame)

            prep_boxes, prep_classes, prep_scores = preproc_data(detections, frame)

            if prep_boxes is not None:
                outputs = self.deepsort.update(prep_boxes, prep_scores, frame)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                draw_bboxes(frame, bbox_xyxy, identities)

            #image_path = 'E:/results/' + str(frame_count) + '.jpg'
            #cv2.imwrite(image_path, frame)

            cv2.imshow('Object detector', frame)
            if cv2.waitKey(1) == ord('q'):
                break



if __name__ == "__main__":
    import sys

    det = Detector()
    det.detect()
