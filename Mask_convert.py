from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np


def Load_img_in_color(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal)
    img == cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


green_img_path = [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Mask/Ethanol red_B1_19h_1.png"
] + [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Mask/Ethanol red_B{}_{}h_{}.jpg".format(
        i[0], i[1], i[2]
    )
    for i in [
        (1, 19, 2),
        (3, 12, 1),
        (3, 12, 2),
        (3, 19, 1),
        (3, 19, 2),
        (2, 17, 1),
        (2, 17, 2),
    ]
]


def Convert_green_mask_to_mask(file_path):
    img = Load_img_in_color(file_path)
    rows, cols, k = img.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    Green = np.array([0, 255, 0])
    White = np.array([255, 255, 255])
    Black = np.array([0, 0, 0])
    for i in range(rows):
        for j in range(cols):
            # print(img[i][j])
            # print(Green)
            # print(type(img[i][j]))
            # print(type(Green))
            if all([abs(img[i][j][z] - Green[z]) < 20 for z in range(3)]):
                img[i][j] = Black
            else:
                img[i][j] = White
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    return img


index = 0
for i in green_img_path:
    Ans = Convert_green_mask_to_mask(i)
    cv2.imwrite(
        "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Mask/Mask{}.png".format(
            index
        ),
        Ans,
    )
    index += 1
