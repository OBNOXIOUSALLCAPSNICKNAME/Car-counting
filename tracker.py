import os
import cv2
import sys
import utils
import json
import time
import math

import numpy as np

from collections import Counter

class Tracker:
    def __init__(self,
        dist_treshold,
        log_length,
        life_length,
        validation_length):

        self.dist_treshold = dist_treshold

        self.log_length = log_length
        self.life_length = life_length
        self.validation_length = validation_length

        self.unique_id = 1

        self.tracks = []
        self.graveyard = []


    def get_tracks(self):
        return self.tracks, self.graveyard

    def reset(self):
        self.tracks = []
        self.graveyard = []
        self.unique_id = 1

    def update(self, boxes):
        self.graveyard = []
        centers = []
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2
            centers.append(np.array([cx, cy]).astype('int'))
        if len(self.tracks) == 0:
            for x in range(len(boxes)):
                self.tracks.append([0, centers[x], boxes[x], ['seen'], [centers[x]]])
        else:
            self.bound_boxes(boxes, centers)
            self.check_presence()

    def update_track(self, id, pos=None, state="lost"):
        if pos is None:
            center = self.tracks[id][4][0]
            bndbox = self.tracks[id][2]
        else:
            bndbox, center = pos
        self.tracks[id][2] = bndbox
        self.tracks[id][3].insert(0, state)
        self.tracks[id][4].insert(0, center)
        if len(self.tracks[id][3]) > self.log_length:
            self.tracks[id][3].pop(self.log_length)
            self.tracks[id][4].pop(self.log_length)

    def check_presence(self):
        valid_tracks = []
        for track in self.tracks:
            if len(track[3]) == self.log_length:
                if track[3][0] == 'lost' and np.all(track[3][0] == np.asarray(track[3])) == True:
                    if track[0] != 0:
                        self.graveyard.append(track.copy())
                else:
                    valid_tracks.append(track.copy())
            elif len(track[3]) == self.validation_length:
                last_state = track[3][-self.validation_length:]
                if np.all('seen' == np.asarray(last_state)) == True:
                    track[0] = self.unique_id
                    valid_tracks.append(track.copy())
                    self.unique_id += 1
            else:
                valid_tracks.append(track.copy())

        self.tracks = valid_tracks.copy()


    def calc_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def bound_boxes(self, boxes, centers, start_time=0):
        prev_detections = [track[4][0] for track in self.tracks]

        #находим расстояние от каждого объекта на предыдущем кадре до каждого объекта на текущем
        all_pair = []
        for i in range(len(prev_detections)):
            for j in range(len(centers)):
                distance = self.calc_distance(prev_detections[i], centers[j])
                all_pair.append([distance, (i, j)])

        if len(all_pair) != 0:

            #для каждого объекта на предыдущем кадре находим ближайший объект на текущем кадре
            unique_prev = np.unique(np.asarray(list(np.asarray(all_pair)[:, 1]))[:, 0])
            groups_prev = [[y for y in all_pair if y[1][0]==x] for x in unique_prev]

            closest_to_prev = []
            for group in groups_prev:
                group.sort()
                if group[0][0] < self.dist_treshold:
                    closest_to_prev.append(group[0])

            #для каждого объекта на текущем кадре находим ближайший объект на предыдущем кадре
            if len(closest_to_prev) != 0:
                unique_current = np.unique(np.asarray(list(np.asarray(closest_to_prev)[:, 1]))[:, 1])
                groups_current = [[y for y in closest_to_prev if y[1][1]==x] for x in unique_current]
            else:
                groups_current = []

            closest_to_current = []
            for group in groups_current:
                group.sort()
                if group[0][0] < self.dist_treshold:
                    closest_to_current.append(group[0])

            #разделяем объекты предыдущего кадра на связанные и свободные
            if len(closest_to_prev) != 0:
                bounded_prev = np.asarray(list(np.asarray(closest_to_current)[:, 1]))[:, 0]
                free_prev = [x for x in range(len(prev_detections)) if x not in bounded_prev]
            else:
                free_prev = [x for x in range(len(prev_detections))]

            #разделяем объекты текущего кадра на связанные и свободные
            if len(closest_to_current) != 0:
                bounded_current = np.asarray(list(np.asarray(closest_to_current)[:, 1]))[:, 1]
                free_current = [x for x in range(len(centers)) if x not in bounded_current]
            else:
                free_current = [x for x in range(len(centers))]

            #обновляем треки, которые были найдены на текущем кадре
            for i in range(len(groups_current)):
                index_1 = closest_to_current[i][1][0]
                index_2 = closest_to_current[i][1][1]

                center = centers[index_2]
                bndbox = boxes[index_2]
                self.update_track(index_1, pos=[bndbox, center], state='seen')

            #обновляем треки, которые не были найденны на текущем кадре
            for i in range(len(free_prev)):
                index = free_prev[i]
                self.update_track(index, state='lost')

            #создаем новые треки для объектов на текущем кадре, для которых не было найденно пары
            for i in range(len(free_current)):
                index = free_current[i]
                center = centers[index]
                bndbox = boxes[index]
                self.tracks.append([0, center, bndbox, ['seen'], [center]])

        else:
            for index in range(len(prev_detections)):
                self.update_track(index, state='lost')
