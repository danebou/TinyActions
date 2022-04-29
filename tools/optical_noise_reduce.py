from turtle import shape
import cv2 as cv
import cv2
import numpy as np
import pickle
import os
import random
import shutil
import json

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

def noise_reduce_frames(frames):
    frames_copy = [cv.cvtColor(f.copy(), cv.COLOR_BGR2HSV) for f in frames]
    final_frames = []
    for i in range(len(frames_copy)):
        p_frame = frames_copy[max(0, i -1)]
        n_frame = frames_copy[min(len(frames_copy) - 1, i +1)]
        frame = frames_copy[i]

        min_thres = 20

        mag1 = frame[..., 2]
        mag1 = np.where(mag1 > min_thres, 1.0, 0)
        mag2 = p_frame[..., 2]
        mag2 = np.where(mag2 > min_thres, 1.0, 0)
        mag3 = n_frame[..., 2]
        mag3 = np.where(mag3 > min_thres, 1.0, 0)

        kernel1 = np.array([[1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])
        kernel2 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])
        identity1 = cv2.filter2D(src=mag1, ddepth=-1, kernel=kernel1).astype(np.float32)
        identity2 = cv2.filter2D(src=mag2, ddepth=-1, kernel=kernel2).astype(np.float32)
        identity3 = cv2.filter2D(src=mag3, ddepth=-1, kernel=kernel2).astype(np.float32)
        identity = identity1 + identity2 + identity3
        #identity = np.expand_dims(identity, 2)

        the_frame = frame.copy()
        the_frame[..., 2] = np.where(identity > 1 * 3 * 2, frame[..., 2], 0).astype(np.uint8)
        the_frame = cv.cvtColor(the_frame, cv.COLOR_HSV2BGR)

        show_frame = frames[i]
        show_frame2 = the_frame
        #cv2.imshow("s",  cv2.resize(show_frame, (512, 512), cv2.INTER_CUBIC))
        #cv2.imshow("s2",  cv2.resize(show_frame2, (512, 512), cv2.INTER_CUBIC))

        #cv2.waitKey(50)
        final_frames.append(the_frame)

    return final_frames


virat_folder = 'optical_flow_stab\\'
virat_files = [(f, os.path.join(dp, f)) for dp, dn, fn in os.walk(os.path.expanduser(virat_folder)) for f in fn]

file_count = len(virat_files)

for i, (file_name, file_path) in enumerate(virat_files):
    if not file_name.endswith(".avi"): continue
    file_rel_path = file_path.replace(virat_folder, "")
    target_file = f'optical_flow_noise_reduce_stab\\{file_rel_path}'
    target_dir = target_file.replace(file_name, "")
    #target_file = target_file.replace('.mp4', '.json')
    print(f'File {i}/{file_count} {target_file}')
    if os.path.isfile(target_file): continue

    os.makedirs(target_dir, exist_ok=True)

    frames = load_video(file_path)
    reduce_frames = noise_reduce_frames(frames)
    assert len(reduce_frames) == len(frames)

    out = cv2.VideoWriter(target_file,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frames[0].shape[0],frames[0].shape[1]))
    for f in reduce_frames:
        out.write(f)
    out.release()
