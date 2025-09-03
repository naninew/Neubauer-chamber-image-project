import cv2
from pathlib import Path
import numpy as np


def Contour(img, Resolution, show_process=False, origin_img=None):
    rows, cols = img.shape
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
    if show_process:
        cv2.rectangle(origin_img, (Min_x, Min_y), (Max_x, Max_y), (255, 255, 0), 5)
    Big_S = (Max_y - Min_y) * (Max_x - Min_x)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    img = ~img
    contours, hierarchy = cv2.findContours(img, 1, 2)
    Big_box = [[None]]
    S_Limit = {"956x1276": (0.2, 0.3), "3024x4032": (0.1, 0.2)}
    MinS_ratio, MaxS_ratio = S_Limit[Resolution]
    Areas = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = cv2.contourArea(box)
        Areas.append(area / Big_S)
        if Big_S * MinS_ratio <= area <= Big_S * MaxS_ratio:
            Big_box = box.copy()
            if show_process:
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.drawContours(origin_img, [box], 0, (255, 0, 0), 5)

                cv2.circle(origin_img, leftmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, rightmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, topmost, 5, (0, 0, 255), -1)
                cv2.circle(origin_img, bottommost, 5, (0, 0, 255), -1)

                # hull = cv2.convexHull(cnt, returnPoints=False)
                # defects = cv2.convexityDefects(cnt, hull)

            #     for i in range(defects.shape[0]):
            #         s, e, f, d = defects[i, 0]
            #         start = tuple(cnt[s][0])
            #         end = tuple(cnt[e][0])
            #         far = tuple(cnt[f][0])
            #         cv2.line(img, start, end, [0, 255, 0], 2)
            #         cv2.circle(img, far, 5, [0, 255, 255], -1)
            break
    # Areas.sort(reverse=True)
    # print(Areas)
    # if show_process:
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(origin_img, cmap=plt.cm.gray)
    #     plt.show()
    return origin_img


def Find_suitable_threshold_ver2(img, Resolution):
    h, bins = np.histogram(img.astype(np.uint8), range(255))
    h[:30] = [0] * 30
    L = 0
    R = 255
    M = (L + R) // 2
    BigSum = sum(h)
    Ratio_dict = {"956x1276": 0.8, "3024x4032": 0.9}
    ratio = Ratio_dict[Resolution]
    while L < M < R:
        SmallSum = sum(h[:M])
        if SmallSum > BigSum * ratio:
            R = M
        else:
            L = M
        M = (L + R) // 2
    return M


def Binary_Threshold(img, threshold=180):
    ret, th2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return th2


def Erode(img, size=5, repeat=1):
    kernel = np.ones((size, size), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(img, kernel, iterations=repeat)
    return img_erosion


def Dilatation(img, size=5, repeat=1):
    kernel = np.ones((size, size), np.uint8)
    img_dilatation = cv2.dilate(img, kernel, iterations=repeat)
    return img_dilatation


def Clear_small_ob(bin_img, Resolution):
    Limit_Areas = {
        "956x1276": 100,
        "3024x4032": 500,
    }
    Min_area = Limit_Areas[Resolution]
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        area = cv2.contourArea(c)
        if area < Min_area:
            cv2.drawContours(bin_img, [c], 0, 0, -1)
            # continue
        # M = cv2.moments(c)
        # In_box = In_which_box(c, Small_boxes)
        # if In_box != None:
        #     Count_yeast[In_box] += 1
    return bin_img


def Load_img_in_color(win_path):
    path_universal = Path(win_path)
    img = cv2.imread(path_universal)
    assert img is not None, "file could not be read, check with os.path.exists()"
    return img


def Process(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # COLOR_BGR2GRAY
    origin_img = Load_img_in_color(path)
    rows, cols = img.shape
    Resolution = sorted([rows, cols])
    Resolution = str(Resolution[0]) + "x" + str(Resolution[1])

    Threshold = Find_suitable_threshold_ver2(img, Resolution)
    img = Binary_Threshold(img, Threshold)
    BigSquareImg = img.copy()
    img = Clear_small_ob(img, Resolution)
    # Find_max_sum_square(img.copy())
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    Enro_Dili_Ratio = {
        "956x1276": ((3, 6), (3, 12), (3, 6)),
        "3024x4032": ((3, 6), (3, 12), (3, 6)),
    }
    ED_ratio = Enro_Dili_Ratio[Resolution]

    for Round in range(1):
        BigSquareImg = Dilatation(
            BigSquareImg,
            size=ED_ratio[0][0] - Round * 2,
            repeat=ED_ratio[0][1] - Round * 2,
        )  # 3,6
        # plt.figure(figsize=(10, 10))
        # plt.imshow(BigSquareImg, cmap=plt.cm.gray)
        # plt.show()
        BigSquareImg = Erode(
            BigSquareImg,
            size=ED_ratio[1][0] - Round * 2,
            repeat=ED_ratio[1][1] - Round * 2,
        )  # 3,12
        # plt.figure(figsize=(10, 10))
        # plt.imshow(BigSquareImg, cmap=plt.cm.gray)
        # plt.show()
        BigSquareImg = Dilatation(
            BigSquareImg,
            size=ED_ratio[2][0] - Round * 2,
            repeat=ED_ratio[2][1] - Round * 2,
        )  # 3,6

    return Contour(
        BigSquareImg,
        Resolution,
        show_process=True,
        origin_img=origin_img.copy(),
    )


Large_input_path_new = tuple(
    [
        "NgocAnh_work/2025_04_20-T8-camIP14/IMG_{}.jpeg".format(i)
        for i in list(range(2022, 2033)) + list(range(2034, 2442))
    ]
    + [
        "NgocAnh_work/2025_04_20-T8-nhuom-camIP14/IMG_{}.jpeg".format(i)
        for i in range(2442, 2545)
    ]
)
for path in Large_input_path_new:
    # Read color image
    image = cv2.imread(path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize grayscale image
    normalized_gray_image = cv2.normalize(
        gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )

    # Convert normalized grayscale image back to color
    normalized_color_image = cv2.cvtColor(normalized_gray_image, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Normalized Image", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Original Image", 700, 700)
    cv2.resizeWindow("Normalized Image", 700, 700)
    # Display original and normalized images
    cv2.imshow("Original Image", image)
    cv2.imshow("Normalized Image", normalized_color_image)
    cv2.waitKey(0)

    image = Process(image, path)
    normalized_color_image = Process(normalized_color_image, path)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Normalized Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 700, 700)
    cv2.resizeWindow("Normalized Image", 700, 700)
    cv2.imshow("Original Image", image)
    cv2.imshow("Normalized Image", normalized_color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
