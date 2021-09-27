import os
import numpy as np
import cv2


C = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (100, 40, 40),
    3: (55, 90, 80),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45, 60, 150),
    22: (145, 170, 100),
}


def render(bev):
    id2color = {
        0: (255, 255, 255),
        1: (0, 0, 255),
        2: (220, 20, 60),
        3: (128, 64, 128),
        4: (70, 70, 70),
        5: (107, 142, 35)
    }
    segmap = np.zeros((256, 256, 3))
    for i in range(256):
        for j in range(256):
            segmap[i, j, :] = id2color[bev[i, j]]
    return segmap

def meta():
    path = '/data_1/yml/V2XSIM/sweeps/BEV_TOP_RAW_id_0/scene_5_000045.jpg'
    from PIL import Image
    im = np.array(Image.open(path))[:, :, 0]
    im[im > 22] = 0
    segmap = np.zeros(list(im.shape)+[3])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            segmap[i, j, :] = C[im[i, j]]
    cv2.imwrite('./complete.png', segmap[:, :, ::-1])

def main():
    train = './dataset/lowerbound/train/agent0/10_0/0.npy'
    test = './dataset/lowerbound/test/agent0/75_0/0.npy'
    # train = '/data1/yimingli/upperbound_seg/dataset/upperbound/train/agent1/10_0/0.npy'
    # test = '/data1/yimingli/upperbound_seg/dataset/upperbound/test/agent0/75_0/0.npy'


    bev_train = np.load(train, allow_pickle=True).item()['bev_seg']
    bev_test = np.load(test, allow_pickle=True).item()['bev_seg']

    seg_train = render(bev_train)
    seg_test = render(bev_test)

    cv2.imwrite('./train.png', seg_train[:, :, ::-1])
    cv2.imwrite('./test.png', seg_test[:, :, ::-1])

if __name__ == '__main__':
    # main()
    meta()
