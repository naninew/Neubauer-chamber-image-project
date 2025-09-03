import cv2
import numpy as np
from pathlib import Path


def Binary_Threshold(img, threshold=180):
    ret, th2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return th2


def Load_img(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Load_img_in_color(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Clear_small_ob(bin_img):
    rows, cols = bin_img.shape
    Min_area = rows * cols // 10
    contours, hierarchy = cv2.findContours(bin_img, 1, 2)
    for c in contours:
        epsilon = 0.001 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, epsilon, True)
        area = cv2.contourArea(c)
        if area < Min_area:
            cv2.drawContours(bin_img, [c], 0, 0, -1)
    return bin_img


def Find_Big_Circle(path):
    img = Load_img(path)
    rows, cols = img.shape
    Threshold = 150
    img = Binary_Threshold(img, Threshold)
    img = Clear_small_ob(img)
    Min_y = 0
    Max_y = rows
    while not any(img[Min_y]):
        Min_y += 1
    while not any(img[Max_y - 1]):
        Max_y -= 1
    Min_x = 0
    Max_x = cols
    while not any(img[Min_y:Max_y, Min_x]):
        Min_x += 1
    while not any(img[Min_y:Max_y, Max_x - 1]):
        Max_x -= 1
    return (Min_x, Max_x, Min_y, Max_y)


# --------------------------------------------------------------------------------------
def Scale_image(path):
    img = Load_img_in_color(path)
    Scale_box = Find_Big_Circle(path)
    Scaled_img = img[Scale_box[2] : Scale_box[3], Scale_box[0] : Scale_box[1]].copy()
    return Scale_box
    # Scaled_img = cv2.resize(Scaled_img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    # return Scaled_img
