from re import L
from turtle import shape
import cv2 as cv
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import  pickle
import matplotlib.pyplot as plt
import random


# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture(f"TinyVIRAT-v2/videos/test/0{random.randint(1000, 5000):4}.mp4")
# 05150
# 00050
# 01250

# Noise: 01050


# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()
o_frame = first_frame
#first_frame = cv2.resize(first_frame, (512, 512))

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

prev_gray = None


frames = []
i = 0
while(cap.isOpened()):
    #if i >= 1: break
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
    i += 1
cap.release()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def get_frame_blob(f):
    blobs = np.arange(f.shape[0] * f.shape[1])
    blobs = blobs.reshape(f.shape[:2])
    kernal = [(-1, -1,), (-1, 0,), (-1, 1,), (0, -1,), (0, 1,), (1, -1,), (1, 0,), (1, 1,)]
    f = cv.blur(f, (3,3))
    kernal_connections = []
    threshold = 10
    for k in kernal:
        shift = np.roll(f, k, (0, 1))
        diff = np.square(f - shift)
        dis = np.sum(diff, axis=2)
        dis = np.clip(dis, 0, 255)
        if k[0] == -1:
            dis[-1, :] = threshold
        elif k[0] == 1:
            dis[0, :] = threshold

        if k[1] == -1:
            dis[:, -1] = threshold
        elif k[1] == 1:
            dis[:, 0] = threshold

        kernal_connections.append(dis < threshold)

    # cv.imshow("dis",  cv2.resize(dis, (512, 512)))
    cv.imshow("f",  cv2.resize(f, (512, 512)))
    cv.waitKey(1000)

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

def happy_mod(a, mod):
    a = a % mod
    if a < 0:
        a += mod
    return a

def frame_diff_pos_mask(img1, img2, pos1, pos2):
    diff = img2.astype(np.float32) - img1.astype(np.float32)
    sx, sy = (pos2[0] - pos1[0], pos2[1] - pos1[1])

    h, w, _ = img1.shape

    x_end = happy_mod(pos2[0], w)
    y_end = happy_mod(pos2[1], h)
    x_start = happy_mod(pos1[0], w)
    y_start = happy_mod(pos1[1], h)

    sx_sign = np.sign(sx)
    sy_sign = np.sign(sy)

    if sx_sign > 0:
        if x_end < x_start:
            diff[0:x_end, :] = 0
            diff[x_start:, :] = 0
        else:
            diff[x_start:x_end, : ] = 0
    elif sx_sign < 0:
        if x_end > x_start:
            diff[0:x_start, :] = 0
            diff[x_end:, :] = 0
        else:
            diff[x_end:x_start, : ] = 0

    if sy_sign > 0:
        if y_end < y_start:
            diff[:, 0:y_end] = 0
            diff[:, y_start:] = 0
        else:
            diff[:, y_start:y_end] = 0
    elif sy_sign < 0:
        if y_end > y_start:
            diff[:, 0:y_start] = 0
            diff[:, y_end:] = 0
        else:
            diff[:, y_end:y_start] = 0

    return diff

while True:
    jitter_value = []
    jitter_pos = []
    pre_proccesed_frames = []
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
        jitter_pos.append(pos)

        f_stable = shift_img(f_blur, pos, True)
        pre_proccesed_frames.append(f_stable)

    for i in range(len(frames)):
        # jitter = np.average(diff_img(cmp_1, cmp_2))
        # jitter_value.append(jitter)
        # jitter_pos.append((last_pos[0] ** 2 + last_pos[1] ** 2) * 20)

        # [0.5, 1.0, 0.5]
        # -     [0.5, 1.0, 0.5] =
        #kernel = [1, 2, -2, -1]
        kernel = [1, 1, -1, -1]
        tmp_frame = np.zeros(pre_proccesed_frames[0].shape, np.float32)
        kernel_sum = 0
        kernel_indices = []
        for j in range(len(kernel)):
            index = j - 1 + i
            if index < 0:
                index = 0
            elif index >= len(frames):
                index = len(frames) -1

            kernel_indices.append(index)
            k_v = kernel[j]
            tmp_frame += pre_proccesed_frames[index] * k_v
            kernel_sum += k_v
        kp0, kp1, kp2, kp3 = kernel_indices
        tmp_frame = (frame_diff_pos_mask(pre_proccesed_frames[kp3], pre_proccesed_frames[kp0], jitter_pos[kp3], jitter_pos[kp0]) +
            frame_diff_pos_mask(pre_proccesed_frames[kp2], pre_proccesed_frames[kp1], jitter_pos[kp2], jitter_pos[kp1]))

        tmp_frame /= 2
        tmp_frame = np.clip(tmp_frame, 0, 255)
        #tmp_frame = tmp_frame.astype(np.uint8)

        cv.imshow("ssadff",  cv2.resize(tmp_frame, (512, 512), cv.INTER_CUBIC))

        if p_f is None:
            p_f = pre_proccesed_frames[i]

        # kernel1 = np.array([[-1, 0, 1],
        #             [-2, 0, 2],
        #             [-1, 0, 1]])
        # identity = cv2.filter2D(src=tmp_frame, ddepth=-1, kernel=kernel1)

        # f_p = np.array(frames[i+1])
        # f_p = cv.blur(cv.blur(f_blur, (3, 3)), (3, 3))
        #diff = tmp_frame.astype(np.float32) - p_f
        # diff = tmp_frame.astype(np.float32)
        # diff = np.square(diff)
        # dis = np.sum(diff, axis=2)
        dis = tmp_frame
        dis = np.square(dis)
        dis = np.sum(dis, axis=2)
        dis = np.sqrt(dis)
        dis = dis * 2
        #dis += 127
        dis = np.clip(dis, 0, 255)
        dis = dis.astype(np.uint8)

        hsv = np.zeros_like(frames[i])
        hsv[..., 1] = 255


        flow = cv.calcOpticalFlowFarneback(cv.cvtColor(pre_proccesed_frames[i], cv.COLOR_BGR2GRAY), cv.cvtColor(p_f, cv.COLOR_BGR2GRAY), None, 0.5, 3,winsize=5, iterations=3, poly_n=5, poly_sigma=1.05, flags=0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = dis #cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #bgr = shift_img(bgr, (-jitter_pos[i][0], -jitter_pos[i][1]))
        cv.imshow("this",  cv2.resize(bgr, (512, 512), cv.INTER_CUBIC))
        # flow = cv.calcOpticalFlowFarneback(cv.cvtColor(tmp_frame, cv.COLOR_BGR2GRAY), cv.cvtColor(p_f, cv.COLOR_BGR2GRAY), None, 0.5, 3,winsize=3, iterations=3, poly_n=3, poly_sigma=1.05, flags=0)
        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv = hsv.copy()
        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 2] = mag # cv.normalize(dis, None, 0, 255, cv.NORM_MINMAX)
        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imshow("fdfddffdfd",  cv2.resize(bgr, (512, 512), cv.INTER_CUBIC))

        # edges = cv2.Canny(image=f, threshold1=50, threshold2=300)

        # cv.imshow("s",  cv2.resize(edges, (512, 512), cv.INTER_CUBIC))
        cv.imshow("f",  cv2.resize(dis, (512, 512), cv.INTER_CUBIC))
        # cv.imshow("a",  cv2.resize(tmp_frame, (512, 512), cv.INTER_CUBIC))
        cv.imshow("s",  cv2.resize(shift_img(frames[i], jitter_pos[i]), (512, 512), cv.INTER_CUBIC))

        cv.waitKey(50)

        p_f = pre_proccesed_frames[i]
    print('done')
    exit()
    plt.plot(jitter_value)
    plt.plot(jitter_pos)
    plt.show()





for f in frames:
    blob = get_frame_blob(f)

exit()






for f in frames:
    #f = cv.blur(f, (1,1))
    f_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
    f_hsv = np.array(f_hsv)
    f = np.array(f)
    def get_points(f):
        points = []
        for x in range(f.shape[0]):
            for y in range(f.shape[1]):
                p = f_hsv[x, y]
                b, g, r = p
                points.append([x, y, b, g, r])
        return np.array(points)

    metric_weights = [1, 1, 0, 0, 1/3]
    def CustomMetric(x, y):
        diff = (y - x) ** 2
        return np.dot(diff, metric_weights)

    new_image = np.zeros(f.shape, np.uint8)
    # frames_points = get_points(f)
    # label_colors = {}
    # clustering = DBSCAN(eps=15, min_samples=8,metric=CustomMetric).fit(frames_points)
    # for i in range(len(clustering.labels_)):
    #     x, y, _, _, _ = frames_points[i]
    #     label = clustering.labels_[i]
    #     if (label == -1): continue
    #     if label not in label_colors:
    #         label_colors[label] = np.random.choice(range(256), size=3)

    #     new_image[x, y] = label_colors[label]

    g = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    # new_image = cv2.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,3,5)

    blur = cv.GaussianBlur(g,(5,5),0)
    _,new_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # while True:
    cv.imshow("dense optical flow", cv2.resize(new_image, (512, 512)))
    cv.imshow("f",  cv2.resize(f, (512, 512), cv.INTER_NEAREST))
    cv.waitKey(1000)

exit()


















frames = np.array(frames)

def get_points(f):
    points = []
    for t in range(f.shape[0]):
        for x in range(f.shape[1]):
            for y in range(f.shape[2]):
                p = f[t, x, y]
                b, g, r = p
                points.append([t, x, y, b, g, r])
    return np.array(points)

metric_weights = [1, 1, 1, 1/5, 0, 0]
def CustomMetric(x, y):
    diff = (y - x) ** 2
    return np.dot(diff, metric_weights)

new_image = np.zeros(frames.shape, np.uint8)
frames_points = get_points(frames)
label_colors = {}
clustering = DBSCAN(eps=1, min_samples=10,metric=CustomMetric).fit(frames_points)
for i in range(len(clustering.labels_)):
    t, x, y, _, _, _ = frames_points[i]
    label = clustering.labels_[i]
    if (label == -1): continue
    if label not in label_colors:
        label_colors[label] = np.random.choice(range(256), size=3)

    new_image[t, x, y] = label_colors[label]

while True:
    for i in range(len(frames)):
        img = cv.resize(frames[i], (512, 512))
        cv.imshow("input", img)
        img2 = cv.resize(new_image[i], (512, 512))
        cv.imshow("input2", img2)
        cv.waitKey(100)

exit()

while(cap.isOpened()):
    prev_frame = o_frame

    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, o_frame = cap.read()

    # Opens a new window and displays the input
    # frame


    # Converts each frame to grayscale - we previously
    # only converted the first frame to grayscale
    frame = o_frame # cv2.resize(o_frame, (512, 512))
    gray = (np.float32(cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)) + np.float32(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))) / 2
    gray = np.uint8(gray)
    #gray = frame[..., 1]
    gray = cv.blur(gray, (1,1))
    #_, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    gray_show = cv2.resize(gray, (500,500), cv.INTER_NEAREST)
    cv.imshow("input", gray_show)

    if prev_gray is None:
        prev_gray = gray

    cv.imshow("input2", cv2.resize(cv.absdiff(o_frame, prev_frame)*10, (512,512), cv.INTER_NEAREST))

    #params = cv2.SimpleBlobDetector_Params()

    #params.filterByArea = False
    #params.minArea = 10

    #detector = cv2.SimpleBlobDetector_create(params)
    #keypoints = detector.detect(gray)

    # Draw detected blobs as red circles.

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

    #blank = np.zeros((1, 1))
    #blobs = cv2.drawKeypoints(gray, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #blobs_show = cv2.resize(blobs, (500,500), cv.INTER_NEAREST)
    #cv2.imshow("Blobs Using Area", blobs_show)




    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 1, 31, 9, 7, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN )

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Opens a new window and displays the output frame
    rgb_show = cv2.resize(rgb, (500,500), cv2.INTER_NEAREST)
    cv.imshow("dense optical flow", rgb_show)

    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()
