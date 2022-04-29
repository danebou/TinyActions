from re import L
from turtle import shape
import cv2 as cv
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import  pickle
import matplotlib.pyplot as plt
import random
import json

from sklearn.utils import shuffle

def load_video(filename):
    cap = cv.VideoCapture(filename)
    frames = []
    i = 0
    while(cap.isOpened()):
        #if i >= 1: break
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        i += 1
    cap.release()
    return frames

import os
from random import shuffle

virat_folder = 'C:\\Users\\dbouchie\\Downloads\\TinyActions-main\\TinyActions-main\\TinyVIRAT-v2\\videos'
pose_folder = 'C:\\Users\\dbouchie\\Downloads\\TinyActions-main\\TinyActions-main\\TinyVIRAT-v2\\pose_high_confidence'
virat_files = [(f, os.path.join(dp, f)) for dp, dn, fn in os.walk(os.path.expanduser(virat_folder)) for f in fn]
shuffle(virat_files)

file_count = len(virat_files)

from tqdm import tqdm

for i, (file_name, file_path) in enumerate(virat_files):
    if not file_name.endswith(".mp4"): continue
    file_rel_path = file_path.replace(virat_folder, "")
    print(f'File {i}/{file_count} {file_rel_path}')
    pose_file = pose_folder + file_rel_path.replace('.mp4', '.json')

    with open(pose_file, 'r') as f:
        pose_json = json.load(f)

    pose_data = {}
    for p in pose_json:
        id = int(p['image_id'][:-len('.jpg')])
        if not id in pose_data:
            pose_data[id] = []
        pose_data[id].append(p)

    for k,v in pose_data.items():
        pose_data[k] = sorted(v, key = lambda x: x['score'], reverse=True)[:5]


    frames = load_video(file_path)

    for i,f in enumerate(frames):
        if i in pose_data:
            for p in pose_data[i]:
                kp = p['keypoints']
                for j in range(len(kp) // 3):
                    cv.circle(f, (int(kp[3*j]), int(kp[3*j+1])), 3, (min(255* kp[3*j+2]*p['score'], 255), 0, 0), 1)

        cv.imshow('main', cv.resize(f, (512,512)))
        cv.waitKey(50)
