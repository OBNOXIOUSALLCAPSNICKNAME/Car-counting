import cv2

import numpy as np
import PIL.ImageColor as ImageColor

from PIL import Image, ImageDraw


STANDARD_COLORS = [
    'White', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet', 'DarkOrchid',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'AliceBlue', 'Chartreuse', 'Aqua', 'LightSteelBlue','ForestGreen', 'Fuchsia',
    'Azure', 'Beige', 'Bisque', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'Aquamarine', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'Gainsboro'
]

def get_index(num):
    if num < len(STANDARD_COLORS):
        return num
    else:
        return num % len(STANDARD_COLORS)


def preproc_bbox(bbox):
    top, right, bottom, left = bbox
    y2 = left
    x2 = top
    y1 = right
    x1 = bottom
    return [x1, y1, x2, y2]

def preproc_boxes(boxes):
    prep_boxes = []
    for bbox in boxes:
        prep_boxes.append(preproc_bbox(bbox))
    return prep_boxes


def scale_bbox(bbox, old_dims, new_dims):
    scale_x = new_dims[0] / old_dims[0]
    scale_y = new_dims[1] / old_dims[1]

    bbox[0] = int(scale_x * bbox[0])
    bbox[1] = int(scale_y * bbox[1])
    bbox[2] = int(scale_x * bbox[2])
    bbox[3] = int(scale_y * bbox[3])

    return bbox

def scale_boxes(tracks, old_dims, new_dims):
    for x in range(len(tracks)):
        tracks[x][5][0] = scale_bbox(tracks[x][5][0], old_dims, new_dims)
    return tracks


def scale_mask(face_landmarks, old_dims, new_dims):
    scale_x = new_dims[0] / old_dims[0]
    scale_y = new_dims[1] / old_dims[1]

    for facial_feature in face_landmarks.keys():
        for i in range(len(face_landmarks[facial_feature])):
            x, y = face_landmarks[facial_feature][i]
            face_landmarks[facial_feature][i] = (int(scale_x * x), int(scale_y * y))

    return face_landmarks

def scale_masks(faces_landmarks, old_dims, new_dims):
    for x in range(len(faces_landmarks)):
        faces_landmarks[x] = scale_mask(faces_landmarks[x], old_dims, new_dims)
    return faces_landmarks


def scale_image(image, new_dims, keep_ar=True):
    if keep_ar == False:
        return cv2.resize(image, (new_dims[0], new_dims[1])), [1, 1], [0, 0]
    else:
        result = np.full((new_dims[1], new_dims[0], 3), [128, 128, 128], dtype=np.uint8)

        new_h = new_dims[1]
        new_w = new_dims[0]

        ar = float(new_dims[1])/float(new_dims[0])
        (h, w) = image.shape[:2]
        offset_x = 0
        offset_y = 0
        if ar < float(h)/float(w):
            scale = new_h / float(h)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            (h, w) = image.shape[:2]
            offset_x = int((new_w - w) / 2)
        else:
            scale = new_w / float(w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            (h, w) = image.shape[:2]
            offset_y = int((new_h - h) / 2)
        return image, [scale, scale], [offset_x, offset_y]


def visualize_bbox(image, coord, status, id, padding):
    ymax, xmax, ymin, xmin = coord

    ymax += padding
    xmax += padding
    ymin -= padding
    xmin -= padding

    length = int((xmax - xmin) / 4.0)

    if status == "seen" and id != 0:
        overlay = image.copy()
        rgb = ImageColor.getrgb(STANDARD_COLORS[get_index(id)])

        cv2.line(overlay, (xmin, ymin), (xmin + length, ymin), rgb, 4)
        cv2.line(overlay, (xmax - length, ymin), (xmax, ymin), rgb, 4)

        cv2.line(overlay, (xmin, ymax), (xmin + length, ymax), rgb, 4)
        cv2.line(overlay, (xmax - length, ymax), (xmax, ymax), rgb, 4)

        cv2.line(overlay, (xmin, ymin), (xmin, ymin + length), rgb, 4)
        cv2.line(overlay, (xmin, ymax - length), (xmin, ymax), rgb, 4)

        cv2.line(overlay, (xmax, ymin), (xmax, ymin + length), rgb, 4)
        cv2.line(overlay, (xmax, ymax - length), (xmax, ymax), rgb, 4)

        image = cv2.addWeighted(overlay, 0.5, image, 1 - 0.5, 0)

    return image

def visualize_label(image, coord, status, id, name, padding):
    ymax, xmax, ymin, xmin = coord

    ymax += padding
    xmax += padding
    ymin -= padding
    xmin -= padding

    coord_y  = ymin-45 if ymin > 50 else ymax + 10

    rgb = ImageColor.getrgb(STANDARD_COLORS[get_index(id)])

    if status == "seen" and id != 0:

        overlay = image.copy()
        rgb = ImageColor.getrgb(STANDARD_COLORS[get_index(id)])
        cv2.rectangle(overlay, (xmin-1, coord_y-1), (xmax+1, coord_y+35), rgb, cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.5, image, 1 - 0.5, 0)
        cv2.rectangle(image, (xmin-1, coord_y-1), (xmax+1, coord_y+35), rgb, 2)

        image = cv2.putText(image,
                            str(id),
                            (xmin,coord_y+27),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (28,25,21), 2, cv2.LINE_AA)

        image = cv2.putText(image,
                            name,
                            (xmin + 50,coord_y+27),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (28,25,21), 2, cv2.LINE_AA)

    return image

def visualize_mask(image, face_landmarks, padding):
    for facial_feature in face_landmarks.keys():
        for i in range(len(face_landmarks[facial_feature]) - 1):
            x1, y1 = face_landmarks[facial_feature][i]
            x2, y2 = face_landmarks[facial_feature][i+1]

            cv2.line(image, (x1, y1), (x2, y2), (215, 215, 65), thickness=1)
            cv2.line(image, (x1, y1), (x1, y1), (215, 215, 65), thickness=3)
            cv2.line(image, (x2, y2), (x2, y2), (215, 215, 65), thickness=3)

    return image


def visualize_tracks(image, tracks):
    for track in tracks:
        xmin, ymin, xmax, ymax = track[2]
        cx, cy = track[4][0]
        rgb = ImageColor.getrgb(STANDARD_COLORS[get_index(track[0])])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), rgb, 2)
        image = cv2.putText(
        image, str(track[0]), (cx , cy),
        cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2, cv2.LINE_AA)

def visualize_crossroad_statistic(image, map):
    h, w, c = image.shape
    image[0:h, 0:230] = np.zeros(c, dtype=int)
    y = 0
    for name, count in map.items():
        y += 50
        image = cv2.putText(
        image, name, (20 , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
        image = cv2.putText(
        image, str(count), (180 , y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
