import os, sys, time, datetime, random, cv2, argparse
import numpy as np

from models import *
from utils.datasets import *
from utils.utils import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms as transforms

from tracker import Tracker
from loader import DataLoader
from processing import*
from vis_utils import*

import PIL.ImageColor as ImageColor


class Track:
    def __init__(self):
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else '')
        self.model = Darknet(opt.config, img_size=opt.img_size)
        self.model.load_state_dict(torch.load(opt.weights, map_location=self.device)['model'])
        self.model.to(self.device).eval()
        self.classes = load_classes(opt.classes)

        self.dataloader = DataLoader(opt.input, opt.output)
        self.preprocessor = Preprocessor()
        self.directions = Directions()
        self.tracker = Tracker(150, 20, 8, 2)

    def run(self):
        for img0, crossroad, n in self.dataloader:
            if n == 1:
                self.tracker.reset()
                self.directions.reset()

            img = self.preprocessor.process(img0, crossroad)
            boxes = self.detect_image(img)
            boxes = self.preprocessor.transform_to_original(boxes)
            self.preprocessor.delete_cache()

            self.tracker.update(boxes)
            tracks, graveyard = self.tracker.get_tracks()
            self.directions.determine_direction(graveyard, crossroad)
            visualize_crossroad_statistic(img0, self.directions.COUNTER_MAP)
            visualize_tracks(img0, tracks)

            if opt.save:
                self.dataloader.save_results(img0)
            if opt.show:
                cv2.imshow('result', img0)
                if cv2.waitKey(1) == ord('q'):
                    break

    def detect_image(self, img):
        im0 = img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[i for i in range(len(opt.classes))])

        boxes = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    boxes.append(np.array(xyxy).astype('int'))
        boxes = np.array(boxes)
        boxes = NMS(boxes, 0.35, return_indices=False)
        return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=992, help='size of processed image. [0,0] for original')
    parser.add_argument('--conf_thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--overlap-thres', type=float, default=0.3, help='IOU threshold for NMS')

    parser.add_argument('--save', action='store_false', help='save displayed results')
    parser.add_argument('--show', action='store_false', help='display results')
    parser.add_argument('--input', type=str, default='E:/Car-counting/input/video/', help='input folder')
    parser.add_argument('--output', type=str, default='E:/Car-counting/output/', help='output folder')

    parser.add_argument('--config', type=str, default='E:/Car-counting/datasets/cars/yolov3.cfg')
    parser.add_argument('--classes', type=str, default='E:/Car-counting/datasets/cars/classes.names')
    parser.add_argument('--weights', type=str, default='E:/Car-counting/weights/best.pt')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        m = Track()
        m.run()
