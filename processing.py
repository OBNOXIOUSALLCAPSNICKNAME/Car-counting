import cv2
import numpy as np
from collections import OrderedDict
from itertools import permutations

from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

class Preprocessor():
    def __init__(self):
        #oktrev [x=295, y=265, w=1335, h=360]
        #gavrilova [x=445, y=435, w=1060, h=330]
        #lenina [x=670, y=295, w=1010, h=330]

        self.cache = OrderedDict()

        self.oktrev_funcs = OrderedDict([
          ("crop",
          {
            "top": 265,
            "left": 295,
            "bottom": 455,
            "right": 290
          }),
          ("resize_pad",
          {
            "width": 992,
            "height": 288
          })
        ])
        self.gavrilova_funcs = OrderedDict([
          ("crop",
          {
            "top": 435,
            "left": 445,
            "bottom": 315,
            "right": 415
          }),
          ("resize_pad",
          {
            "width": 992,
            "height": 288
          })
        ])
        self.lenina_funcs = OrderedDict([
          ("crop",
          {
            "top": 295,
            "left": 670,
            "bottom": 455,
            "right": 240
          }),
          ("resize_pad",
          {
            "width": 992,
            "height": 288
          })
        ])

        self.ANNOT_FUNC_MAP = {
            'annot_scale': self.annot_scale,
            'annot_offset': self.annot_offset
        }
        self.IMAGE_FUNC_MAP = {
            'resize_pad': self.resize_pad,
            'crop': self.crop
        }
        self.CROSSROADS_MAP = {
            'oktrev': self.oktrev_funcs,
            'gavrilova': self.gavrilova_funcs,
            'lenina': self.lenina_funcs
        }


    def _write_to_cache(self, name, **args):
        self.cache[str(len(self.cache) + 1)] = (name, args)


    def delete_cache(self):
        self.cache = OrderedDict()

    def transform_to_original(self, boxes=None, masks=None, landmarks=None):
        for func in reversed(self.cache.items()):
            name, args = func[1]
            boxes = self.ANNOT_FUNC_MAP[name](boxes=boxes, **args)
        return boxes

    def process(self, img, crossroads):
        if crossroads not in self.CROSSROADS_MAP:
            return img
        for name, args in self.CROSSROADS_MAP[crossroads].items():
            img = self.IMAGE_FUNC_MAP[name](img=img, **args)
        return img


    def annot_scale(self, scale_x, scale_y, boxes):
        if boxes.size != 0:
            boxes = boxes.astype("float")
            boxes[:,0] *= scale_x
            boxes[:,1] *= scale_y
            boxes[:,2] *= scale_x
            boxes[:,3] *= scale_y
            boxes = boxes.astype("int")
        return boxes

    def annot_offset(self, offset_x, offset_y, boxes):
        if boxes.size != 0:
            boxes[:,0] += offset_x
            boxes[:,1] += offset_y
            boxes[:,2] += offset_x
            boxes[:,3] += offset_y
        return boxes


    def crop(self, img, top, left, bottom, right):
        h, w = img.shape[:2]
        res = img[top:h-bottom, left:w-right]

        self._write_to_cache(
        'annot_offset',
        offset_x=left, offset_y=top)

        return res

    def resize_pad(self, img, width, height):
        h, w = img.shape[:2]
        new_ar = height / width
        old_ar = h / w
        offset_x = 0
        offset_y = 0
        if new_ar < old_ar:
            scale = height / h
            new_w = int(w * scale)
            offset_x = int((width - new_w) / 2)
        else:
            scale = width / w
            new_h = int(h * scale)
            offset_y = int((height - new_h) / 2)

        img = cv2.resize(img, None, fx=scale, fy=scale)
        (h, w) = img.shape[:2]
        res = np.full((height, width, 3), [128, 128, 128], dtype=np.uint8)
        res[offset_y:offset_y+h, offset_x:offset_x+w] = img

        self._write_to_cache(
        'annot_offset',
        offset_x=-1*offset_x, offset_y=-1*offset_y)
        self._write_to_cache(
        'annot_scale',
        scale_x=1./scale, scale_y=1./scale)

        return res

def NMS(boxes, thres, return_indices=False):
    if boxes is None or boxes.shape[:1][0] == 0:
        if return_indices:
            return boxes, []
        else:
            return boxes

    if boxes.dtype.kind == "i":
    	boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
    	last = len(idxs) - 1
    	i = idxs[last]
    	pick.append(i)

    	xx1 = np.maximum(x1[i], x1[idxs[:last]])
    	yy1 = np.maximum(y1[i], y1[idxs[:last]])
    	xx2 = np.minimum(x2[i], x2[idxs[:last]])
    	yy2 = np.minimum(y2[i], y2[idxs[:last]])

    	w = np.maximum(0, xx2 - xx1 + 1)
    	h = np.maximum(0, yy2 - yy1 + 1)

    	overlap = (w * h) / area[idxs[:last]]

    	idxs = np.delete(idxs, np.concatenate(([last],
    		np.where(overlap > thres)[0])))

    if return_indices:
        return boxes[pick].astype("int"), pick
    else:
        return boxes[pick].astype("int")

class Directions():
    def __init__(self):
        self.oktrev_lines = {
            "top": [np.array([390, 280]), np.array([900, 255])],
            "left": [np.array([242, 422]), np.array([375, 585])],
            "bottom": [np.array([785, 695]), np.array([1360, 675])],
            "right": [np.array([1530, 435]), np.array([1395, 345])]
        }
        self.lenina_lines = {
            "top": [np.array([800, 300]), np.array([910, 290])],
            "left": [np.array([645, 345]), np.array([710, 535])],
            "bottom": [np.array([1235, 625]), np.array([1690, 505])],
            "right": [np.array([1630, 370]), np.array([1485, 325])]
        }
        self.gavrilova_lines = {
            "top": [np.array([10, 10]), np.array([10, 10])],
            "left": [np.array([455, 565]), np.array([645, 750])],
            "bottom": [np.array([1100, 720]), np.array([1300, 650])],
            "right": [np.array([1360, 500]), np.array([1225, 465])]
        }

        self.CROSSROADS_MAP = {
            'oktrev': self.oktrev_lines,
            'lenina': self.lenina_lines,
            'gavrilova': self.gavrilova_lines
        }
        self.reset()

    def reset(self):
        self.COUNTER_MAP = {}
        for a, b in permutations(['top', 'left', 'bottom', 'right'], 2):
            self.COUNTER_MAP['{}-{}'.format(a, b)] = 0


    def distance(self, P, crossroad, ignore=''):
        distances = {}
        for name, l in self.CROSSROADS_MAP[crossroad].items():
            A = l[0]
            B = l[1]
            if name != ignore:
                if all(A == P) or all(B == P):
                    dist = 0
                elif arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
                    dist = norm(P - A)
                elif arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
                    dist = norm(P - B)
                else:
                    dist = norm(cross(A-B, A-P))/norm(B-A)
                distances[name] = dist
        return min(distances, key=distances.get)

    def determine_direction(self, tracks, crossroad):
        for track in tracks:
            start = track[1]
            end = track[4][0]
            a = self.distance(start, crossroad)
            b = self.distance(end, crossroad, a)
            self.COUNTER_MAP['{}-{}'.format(a, b)] += 1
            print(track[0], a, b)
