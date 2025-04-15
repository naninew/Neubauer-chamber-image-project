import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def Load_img(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Merge_img(right_img_path, left_img_path):
    right_img = Load_img(right_img_path)
    left_img = Load_img(left_img_path)
    merge_img = np.hstack((right_img, left_img))
    plt.figure(figsize=(10, 10))
    plt.imshow(merge_img, cmap=plt.cm.gray)
    plt.show()
    return merge_img
