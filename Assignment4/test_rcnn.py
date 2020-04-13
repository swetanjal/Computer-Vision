import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import sys
import xmltodict
import os

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


train_on_gpu = torch.cuda.is_available()
# train_on_gpu = False
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
from torchvision import datasets
import torchvision.transforms as transforms

preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # pool / 2
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        # pool / 2
        self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding = 1)
        # pool / 2
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding = 1)
        # pool / 2
        self.conv7 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)
        # pool / 2
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 20)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))
        # flatten image input
        x = x.view(-1, 512 * 4 * 4)
        # add dropout layer
        # x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        #x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

model = Net()
if train_on_gpu:
    model.cuda()
model.load_state_dict(torch.load('model_voc_60.pt'))

from PIL import Image
def process(image_path):
    bounding_boxes = []
    for i in range(20):
        bounding_boxes.append([])
    img = cv2.imread('./testing/JPEGImages/' + image_path + '.jpg')
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast(base_k = 150, inc_k = 150)
    rects = ss.process()

    for i, rect in enumerate(rects):
        x, y, w, h = rect
        batch = []
        image = Image.open('./testing/JPEGImages/' + image_path + '.jpg')
        cropped_image = image.crop((x, y, x + w, y + h))
        batch.append(preprocess(cropped_image))
        batch = torch.stack(batch)
        if train_on_gpu:
            batch = batch.cuda()
        output = model(batch)
        a, preds_tensor = torch.max(output, 1)
        LABEL = -1
        BBOX = -1
        PR = -1
        if train_on_gpu:
            PR = a.detach().cpu().numpy()[0]
            LABEL = preds_tensor.cpu().numpy()[0]
        else:
            PR = a.detach().numpy()[0]
            LABEL = preds_tensor.numpy()[0]
        if PR >= 0.985 and rect[2] * rect[3] >= 1000 and (LABEL != 19 and LABEL != 7 and LABEL != 2 and LABEL != 4 and LABEL != 3 and LABEL != 12 and LABEL != 8):
            bounding_boxes[LABEL].append([PR, [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]])
        elif PR >= 0.99 and rect[2] * rect[3] >= 1000 and (LABEL == 12 or LABEL == 8):
            bounding_boxes[LABEL].append([PR, [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]])
        elif PR >= 1.0 and rect[2] * rect[3] >= 1000 and (LABEL == 19 or LABEL == 7 or LABEL == 2 or LABEL == 4 or LABEL == 3):
            bounding_boxes[LABEL].append([PR, [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]])
            #cv2.imshow("Output", img)
            #cv2.waitKey()
            #print(LABEL)
            #print(rect)
    LABELS = []
    BBOX = []
    PR = []
    for i in range(20):
        if len(bounding_boxes[i]) <= 3:
            continue
        valid = []
        for j in range(len(bounding_boxes[i])):
            valid.append(1)
        for j in range(len(bounding_boxes[i])):
            if valid[j] == 0:
                continue
            for k in range(len(bounding_boxes[i])):
                if valid[k] == 0 or valid[j] == 0 or j == k:
                    continue
                if IoU(np.array(bounding_boxes[i][j][1]), np.array(bounding_boxes[i][k][1])) >= 0.1:
                    if bounding_boxes[i][j][0] < bounding_boxes[i][k][0]:
                        # print("Rejecting " + str(j))
                        valid[j] = 0
                    else:
                        # print("Rejecting " + str(k))
                        valid[k] = 0
        
        for j in range(len(bounding_boxes[i])):
            if valid[j] == 1:
                img = cv2.rectangle(img, (bounding_boxes[i][j][1][0], bounding_boxes[i][j][1][1]), (bounding_boxes[i][j][1][2], bounding_boxes[i][j][1][3]), (255, 0, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX  
                org = (bounding_boxes[i][j][1][0] + 15, bounding_boxes[i][j][1][1] + 15) 
                fontScale = 0.5
                color = (0, 0, 0) 
                thickness = 1
                txt = classes[i]
                cv2.rectangle(img,(org[0] - 15 + 10, org[1] - 15 + 5),(org[0] + int(len(txt) * 100 / 11), org[1] + 7),(255,255,255),-1) 
                img = cv2.putText(img, txt , org, font, fontScale, color, thickness, cv2.LINE_AA)
                LABELS.append(classes[i])
                BBOX.append([bounding_boxes[i][j][1][0], bounding_boxes[i][j][1][1], bounding_boxes[i][j][1][2], bounding_boxes[i][j][1][3]])
                PR.append(bounding_boxes[i][j][0])
    GT_LABELS = []
    GT_BBOX = []
    with open("./testing/Annotations/" + image_path + ".xml") as fd:
        doc = xmltodict.parse(fd.read())
        try:
            for objects in doc["annotation"]["object"]:
                GT_LABELS.append(objects["name"])
                GT_BBOX.append([int(objects["bndbox"]["xmin"]), int(objects["bndbox"]["ymin"]), int(objects["bndbox"]["xmax"]), int(objects["bndbox"]["ymax"])])
        except:
            objects = doc["annotation"]["object"]
            GT_LABELS.append(objects["name"])
            GT_BBOX.append([int(objects["bndbox"]["xmin"]), int(objects["bndbox"]["ymin"]), int(objects["bndbox"]["xmax"]), int(objects["bndbox"]["ymax"])])
    cv2.imwrite("./Results/" + image_path + ".jpg", img)
    f = open("./groundtruths/" + image_path + ".txt", "w")
    for i in range(len(GT_LABELS)):
        f.write(GT_LABELS[i] + " " + str(GT_BBOX[i][0]) + " " + str(GT_BBOX[i][1]) + " " + str(GT_BBOX[i][2]) + " " + str(GT_BBOX[i][3]) + "\n")
    f.close()
    f = open("./detections/" + image_path + ".txt", "w")    
    for i in range(len(LABELS)):
        f.write(LABELS[i] + " " + str(PR[i]) + " " + str(BBOX[i][0]) + " " + str(BBOX[i][1]) + " " + str(BBOX[i][2]) + " " + str(BBOX[i][3]) + "\n")
    f.close()

files = os.listdir("./testing/Annotations")
files.sort()
c = 1
for f in files:
    print("Done with sample " + str(c))
    c = c + 1
    process(f.replace('.xml', ''))