import torch
import torchvision
import cv2
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from torch import nn
from sklearn.svm import SVC
import os
import xmltodict
import time
import sys

def IoU(box1, box2):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return (base_mat * intersect_ratio)[0]

def storeData(dat, filename):
    dbfile = open(filename, 'wb')
    pickle.dump(dat, dbfile)                      
    dbfile.close()

def loadData(filename):
    dbfile = open(filename, 'rb')
    return pickle.load(dbfile)

def clean_path(path):
    if path.endswith('/'):
        path = path[0: len(path) - 1]
    return path

def showProposals(im, rects):
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()

annotation_folder = './testing/Annotations'
data_folder = './testing/JPEGImages'
train_data = './testing/cnn_data'
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

annotation_folder = clean_path(annotation_folder)
data_folder = clean_path(data_folder)
train_data = clean_path(train_data)
files = os.listdir(annotation_folder)
files.sort()

cnt = 0

samples = 1
for f in files:
    labels = []
    bbox = []
    with open(annotation_folder + "/" + f) as fd:
        doc = xmltodict.parse(fd.read())
        try:
            for objects in doc["annotation"]["object"]:
                labels.append(objects["name"])
                bbox.append([int(objects["bndbox"]["xmin"]), int(objects["bndbox"]["ymin"]), int(objects["bndbox"]["xmax"]), int(objects["bndbox"]["ymax"])])
        except:
            objects = doc["annotation"]["object"]
            labels.append(objects["name"])
            bbox.append([int(objects["bndbox"]["xmin"]), int(objects["bndbox"]["ymin"]), int(objects["bndbox"]["xmax"]), int(objects["bndbox"]["ymax"])])
    img_name = f.replace('.xml', '.jpg')
    for i in range(len(bbox)):
        image = Image.open(data_folder + "/" + img_name)
        cropped_image = image.crop((bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]))
        cropped_image.save(train_data + "/" + labels[i] + "/" + str(cnt) + ".jpg")
        cnt = cnt + 1
    print("Done with Sample " + str(samples))
    samples = samples + 1