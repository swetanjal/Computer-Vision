import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import copy

def convolve(im, k):
    # return signal.convolve2d(im, k, boundary='symm', mode='same')
    im = im.astype(np.float64)
    k = k.astype(np.float64)
    rows = im.shape[0]
    cols = im.shape[1]
    k_r = k.shape[0]
    k_c = k.shape[1]
    res = np.zeros(im.shape)
    for i in range(int(k_r/2), rows - int((k_r-1) / 2)):
        for j in range(int(k_c/2), cols - int((k_c - 1) / 2)):
            l_r = i - int(k_r / 2)
            r_r = i + int((k_r - 1) / 2)
            l_c = j - int(k_c / 2)
            r_c = j + int((k_c - 1) / 2)
            res[i][j] = max(0, sum(sum(k * im[l_r : r_r + 1, l_c : r_c + 1])))
    return res.astype(np.int)

def optical_flow(img1, img2, threshold):
    kernelx = np.array([[-1., 1.], [-1., 1.]])
    kernely = np.array([[-1., -1.], [1., 1.]])
    kernelt = np.array([[1., 1.], [1., 1.]])
    outputx = convolve(img1, kernelx)
    outputy = convolve(img1, kernely)
    outputt = convolve(img2, kernelt) - convolve(img1, kernelt)
    window_size = 15
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    for i in range(int(window_size / 2), img1.shape[0] - int(window_size / 2)):
        for j in range(int(window_size / 2), img1.shape[1] - int(window_size / 2)):
            Ix = outputx[i - int(window_size / 2) : i + int(window_size / 2) + 1, j - int(window_size / 2) : j + int(window_size / 2) + 1].flatten()
            Iy = outputy[i - int(window_size / 2) : i + int(window_size / 2) + 1, j - int(window_size / 2) : j + int(window_size / 2) + 1].flatten()
            It = outputt[i - int(window_size / 2) : i + int(window_size / 2) + 1, j - int(window_size / 2) : j + int(window_size / 2) + 1].flatten()
            A = np.array([[np.sum(Ix * Ix), np.sum(Ix * Iy)], [np.sum(Ix * Iy), np.sum(Iy * Iy)]])
            b = np.array([-np.sum(It * Ix), -np.sum(It * Iy)])
            res = np.linalg.pinv(A).dot(b)
            u[i,j] = res[0]
            v[i,j] = res[1]
    magnitude = np.zeros(img1.shape)
    angle = np.zeros(img1.shape)
    mask = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            magnitude[i, j] = u[i, j] * u[i, j] + v[i, j] * v[i, j]
            angle[i, j] = np.arctan2(v[i, j],u[i, j])
            if magnitude[i, j] >= threshold:
                mask[i, j] = 255
    return outputx, outputy, outputt, u, v, magnitude, angle, mask, \
    'Sobel X', 'Sobel Y', 'Derivative in t', 'Optical Flow in X direction', 'Optical Flow in Y Direction', \
    'Norm', 'Angle', 'Optical Flow Mask'

def process_video(frame_folder, output_folder):
    roi = -1
    files = os.listdir(frame_folder)
    files.sort()
    kp = []
    for i in range(len(files) - 1):
        img1 = cv2.imread(frame_folder + '/' + files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame_folder + '/' + files[i + 1], cv2.IMREAD_GRAYSCALE)
        new_kp = []
        if i == 0:
            #r = cv2.selectROI(img1, fromCenter = False)
            #roi = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
            outputx, outputy, outputt, u, v, magnitude, angle, mask, label1, label2, label3, label4, label5, label6, \
            label7, label8 = optical_flow(img1, img2, 4)
            # kp = cv2.goodFeaturesToTrack(img1, 10000, 0.01, 10)
            # for arrow_ind in kp:
            #     x,y = arrow_ind[0]
            #     y = int(y)
            #     x = int(x)
            #     if x >= roi[0] and x <= roi[2] and y >= roi[1] and y <= roi[3]:
            #         new_kp.append([x + u[y,x],y + v[y,x]])
            #         img1 = cv2.circle(img1, (x, y), 5, (0, 0, 255), 2)
            # kp = copy.deepcopy(new_kp)
            print("Done with a frame!")
            cv2.imwrite(output_folder + '/' + files[i], mask)
        else:
            outputx, outputy, outputt, u, v, magnitude, angle, mask, label1, label2, label3, label4, label5, label6, \
            label7, label8 = optical_flow(img1, img2, 4)
            # for arrow_ind in kp:
            #     x,y = arrow_ind
            #     y = int(y)
            #     x = int(x)
            #     new_kp.append([x + u[y,x],y + v[y,x]])
            #     img1 = cv2.circle(img1, (x, y), 5, (0, 0, 255), 2)
            # kp = copy.deepcopy(new_kp)
            print("Done with a frame!")
            cv2.imwrite(output_folder + '/' + files[i], mask)

process_video('/home/tapas/Desktop/frames', '/home/tapas/Desktop/output')