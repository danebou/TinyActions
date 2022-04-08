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


def get_slid_win_score(img, p_img, shift, ignore_center):
    shift_p_img = np.roll(p_img.copy(), shift, (0, 1))
    diff = diff_img(img, shift_p_img)

    x_shift, y_shift = shift
    if x_shift < 0:
        diff[x_shift:, :] = 0
    elif x_shift > 0:
        diff[:x_shift, :] = 0

    if y_shift < 0:
        diff[:, y_shift:] = 0
    elif x_shift > 0:
        diff[:, :y_shift] = 0

    diff = np.where(diff > 20, 1, 0)

    # avg_x, std_x = weighted_avg_and_std(np.arange(img.shape[0]), np.sum(diff, axis=1))
    # avg_y, std_y = weighted_avg_and_std(np.arange(img.shape[1]), np.sum(diff, axis=0))
    # return std_x ** 2 + std_y ** 2


    # w, h, _ = img.shape
    # cir_pos = (w // 2, h // 2)
    # cir_rad = min(w, h) * 3 // 8
    # mask = np.full(diff.shape, 1)
    # mask = cv2.circle(mask, cir_pos, cir_rad, (0,0,0), -1)
    # diff = np.multiply(diff,mask)

    total_pixels = img.shape[0] * img.shape[1] - abs(x_shift) * img.shape[1] - abs(y_shift) * img.shape[0] + abs(x_shift) * abs(y_shift)
    score = np.sum(diff) / total_pixels

    return score

def shift_img(img, shift, zero=False):
    shift_img = np.roll(img.copy(), shift, (0, 1))
    x_shift, y_shift = shift
    x_shift %= img.shape[0]
    y_shift %= img.shape[1]

    return shift_img


def diff_img(img1, img2):
    diff = np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)
    sqr = np.square(diff)
    dis = np.sum(sqr, axis=2)
    return dis

def uint8_clip(img):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def get_stabalization_data(frames):
    stab = []
    jitter_value = []
    jitter_pos = []
    p_frames = []
    p_frame_len = 5
    p_f = None
    p_stable_frame = None
    w, h, _ = frames[0].shape
    pos = (0,0)
    for i in range(len(frames)):
        f = np.array(frames[i].copy())
        f_blur = cv.blur(f, (3, 3))
        if p_stable_frame is None:
            p_stable_frame = f_blur

        # cir_pos = (w // 2, h // 2)
        # cir_rad = min(w, h) * 3 // 8
        # cmp_1 = cv2.circle(p_frames[0].copy(), cir_pos, cir_rad, (0,0,0), -1)
        # cmp_2 = cv2.circle(p_frames[1].copy(), cir_pos, cir_rad, (0,0,0), -1)

        #cv.imshow("dffdfd",  cv2.resize(uint8_clip(diff_img(cmp_1, cmp_2)), (512, 512), cv.INTER_CUBIC))

        base_kernal = [(0,0), (-1, -1,), (-1, 0,), (-1, 1,), (0, -1,), (0, 1,), (1, -1,), (1, 0,), (1, 1,), ]
        scores = {}
        kernal = base_kernal
        last_pos = (0,0)
        for _ in range(4):
            for b_k in kernal:
                k = (b_k[0] + last_pos[0], b_k[1] + last_pos[1])
                if k in scores: continue
                scores[k] = get_slid_win_score(p_stable_frame, f_blur, k, True)
                #scores[k] = get_slid_win_score(cmp_1, cmp_2, k)
            j = min(scores, key=scores.get)
            if j == last_pos:
                break
            last_pos = j
        p_stable_frame = f_blur.copy()

        pos = (pos[0] + last_pos[0], pos[1] + last_pos[1])
        stab.append(pos)

        f_stable = shift_img(f_blur, pos, True)

        if i < p_frame_len:
            p_frames.append(f_stable)
            continue
        else:
            p_frames = p_frames[1:]
            p_frames.append(f_stable)

        # jitter = np.average(diff_img(cmp_1, cmp_2))
        # jitter_value.append(jitter)
        # jitter_pos.append((last_pos[0] ** 2 + last_pos[1] ** 2) * 20)

        tmp_frame = p_frames[0] * 0.25 + p_frames[1] * 0.75 + p_frames[2] + p_frames[3] * 0.75 + p_frames[4] * 0.25
        tmp_frame /= 3
        tmp_frame = p_frames[0]
        tmp_frame = tmp_frame.astype(np.uint8)

        continue

        if p_f is None:
            p_f = tmp_frame
            continue

        # f_p = np.array(frames[i+1])
        # f_p = cv.blur(cv.blur(f_blur, (3, 3)), (3, 3))
        diff = tmp_frame.astype(np.float32) - p_f
        diff = np.square(diff)
        dis = np.sum(diff, axis=2)
        dis = dis / 2
        #dis += 127
        dis = np.clip(dis, 0, 255)
        dis = dis.astype(np.uint8)

        # edges = cv2.Canny(image=f, threshold1=50, threshold2=300)

        # # cv.imshow("s",  cv2.resize(edges, (512, 512), cv.INTER_CUBIC))
        # cv.imshow("f",  cv2.resize(dis, (512, 512), cv.INTER_CUBIC))
        # # cv.imshow("a",  cv2.resize(tmp_frame, (512, 512), cv.INTER_CUBIC))
        # cv.imshow("s",  cv2.resize(shift_img(frames[i], pos), (512, 512), cv.INTER_CUBIC))

        # cv.imshow("f",  cv2.resize(dis, (512, 512), cv.INTER_CUBIC))
        # # cv.imshow("d",  cv2.resize(cmp_1, (512, 512), cv.INTER_CUBIC))
        # cv.waitKey(50)

        p_f = tmp_frame
    return stab

virat_folder = 'TinyVIRAT-v2\\videos\\'
virat_files = [(f, os.path.join(dp, f)) for dp, dn, fn in os.walk(os.path.expanduser(virat_folder)) for f in fn]

file_count = len(virat_files)

for i, (file_name, file_path) in enumerate(virat_files):
    if not file_name.endswith(".mp4"): continue
    file_rel_path = file_path.replace(virat_folder, "")
    target_file = f'virat_stabilize\\{file_rel_path}'
    target_dir = target_file.replace(file_name, "")
    target_file = target_file.replace('.mp4', '.json')
    print(f'File {i}/{file_count} {target_file}')
    if os.path.isfile(target_file): continue

    os.makedirs(target_dir, exist_ok=True)

    frames = load_video(file_path)
    stab = get_stabalization_data(frames)
    assert len(stab) == len(frames)
    positions_json = {}
    positions_json['x'] = []
    positions_json['y'] = []
    for x,y in stab:
        positions_json['x'].append(x)
        positions_json['y'].append(y)
    with open(target_file, 'w') as f:
        json.dump(positions_json, f)
