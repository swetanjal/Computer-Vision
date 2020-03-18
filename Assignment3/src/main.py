import cv2
import numpy as np
import networkx as nx
from sklearn import mixture
from networkx.algorithms.flow import shortest_augmenting_path

gamma = 1
comp = 5
four_neighbourhood = True
def init_alphas(img, roi):
    # alpha = 1 for foreground
    # alpha = 0 for background
    alpha = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if j >= roi[0] and j <= roi[2] and i >= roi[1] and i <= roi[3]:
                alpha[i][j] = 1
            else:
                alpha[i][j] = 0
    return alpha

def getMixtureModel(D, k):
    clf = mixture.GaussianMixture(n_components = k, covariance_type='full')
    clf.fit(D)
    return clf

def initGMM(img, alphas):
    fg = []
    bg = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if alphas[i][j] == 1:
                fg.append(img[i][j])
            else:
                bg.append(img[i][j])
    bg_GMM = getMixtureModel(bg, comp)
    fg_GMM = getMixtureModel(fg, comp)
    return fg_GMM, bg_GMM

def negative_log_likelihood(d, clf):
    idx = clf.predict(d.reshape(1, -1))[0]
    a = 0.5 * np.log(np.linalg.det(clf.covariances_[idx]))
    b = 0.5 * np.dot(np.dot(np.transpose(d - clf.means_[idx]), np.linalg.inv(clf.covariances_[idx])), d - clf.means_[idx])
    c = -np.log(clf.weights_[idx])
    return (a + b + c)

def getPenalty(img, clf):
    penalty = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            penalty[i][j] = negative_log_likelihood(img[i][j], clf)
    return penalty

def getBinaryPotential(a, b, beta):
    return gamma * np.exp(-1.0 * beta * np.sum((a - b) ** 2))

def createGraph(img, fg_penalty, bg_penalty, alphas):
    G = nx.Graph()
    G.add_node('s')
    G.add_node('t')
    beta = 0.0
    cnt = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            G.add_node((i, j))
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if abs(k) + abs(l) == 0 or (abs(k) + abs(l) > 1 and four_neighbourhood == True):
                        # Not a 4 neighbour
                        # Center
                        continue
                    if (i + k) < 0 or (i + k) >= img.shape[0] or (j + l) < 0 or (j + l) >= img.shape[1]:
                        continue
                    beta = beta + np.sum((img[i][j] - img[i + k][j + l])** 2)
                    cnt = cnt + 1
    beta = (beta * 0.5) / cnt
    inf = 1000000000000000000000000000000000000
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if alphas[i][j] == 0:
                G.add_edge('s', (i, j), capacity = 0)
                G.add_edge((i, j), 't', capacity = inf)
            else:
                G.add_edge('s', (i, j), capacity = bg_penalty[i][j])
                G.add_edge((i, j), 't', capacity = fg_penalty[i][j])
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if abs(k) + abs(l) == 0 or (abs(k) + abs(l) > 1 and four_neighbourhood == True):
                        # Not a 4 neighbour
                        # Center
                        continue
                    if (i + k) < 0 or (i + k) >= img.shape[0] or (j + l) < 0 or (j + l) >= img.shape[1]:
                        continue
                    # Need to add pairwise edges
                    G.add_edge((i, j), (i + k, j + l), capacity = getBinaryPotential(img[i][j], img[i + k][j + l], beta))
    return G

def main(img_name, GAMMA = 1, COMP = 5, ITR = 16, space = 'rgb', four_connectivity = True):
    global four_neighbourhood
    if four_connectivity == False:
        four_neighbourhood = False
    else:
        four_neighbourhood = True
    global gamma
    gamma = GAMMA
    global comp
    comp = COMP
    img = cv2.imread(img_name)
    # f = open('../input_data/bboxes/' + img_name + '.txt', 'r')
    # f = f.readlines()
    # roi = []
    # for _ in f:
    #     _ = _.split()
    #     roi = [int(_[0]), int(_[1]), int(_[2]), int(_[3])]
    #print(roi)
    r = cv2.selectROI(img, fromCenter = False)
    roi = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
    #print(roi)
    if space == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif space == 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif space == 'ycr_cb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    else:
        assert(False)
    roi_img = np.zeros(img.shape)
    alphas = init_alphas(img, roi)
    itr = 0
    while itr < ITR:
        itr = itr + 1
        print("Iteration " + str(itr))
        fg_GMM, bg_GMM = initGMM(img, alphas)
        fg_penalty = getPenalty(img, fg_GMM)
        bg_penalty = getPenalty(img, bg_GMM)
        G = createGraph(img, fg_penalty, bg_penalty, alphas)
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=shortest_augmenting_path)
        reachable, non_reachable = partition
        alphas = np.zeros((img.shape[0], img.shape[1]))
        for px in reachable:
            if px != 's':
                alphas[px[0]][px[1]] = 1
        roi_img = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if alphas[i][j] == 1:
                    roi_img[i][j] = img[i][j]
        roi_img = roi_img.astype(np.uint8)
    if space == 'rgb':
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
    elif space == 'lab':
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_LAB2BGR)
    elif space == 'ycr_cb':
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow('Segmented Image', roi_img)
    cv2.waitKey()

main('../input_data/images/elefant.jpg', GAMMA = 50, space = 'rgb', ITR = 5, four_connectivity = True)