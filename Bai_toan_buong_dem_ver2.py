import cv2
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
import math


# -------------------------------------------------------------
def Show_Histogram_Board(img):
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(max(histg[100:]))
    plt.plot(histg)
    plt.show(block=False)


def Find_suitable_threshold_ver2(img):
    h, bins = np.histogram(img.astype(np.uint8), range(255))
    h[:30] = [0] * 30
    L = 0
    R = 255
    M = (L + R) // 2
    BigSum = sum(h)
    while L < M < R:
        SmallSum = sum(h[:M])
        if SmallSum > BigSum * 0.7:
            R = M
        else:
            L = M
        M = (L + R) // 2
    return M


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


def Contour(img, show_process=False, origin_img=None):
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
    # if show_process:
    #     cv2.rectangle(origin_img, (Min_x, Min_y), (Max_x, Max_y), (255, 255, 0), 5)
    Big_S = (Max_y - Min_y) * (Max_x - Min_x)
    img = ~img
    contours, hierarchy = cv2.findContours(img, 1, 2)
    Big_box = [[None]]
    Areas = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = cv2.contourArea(box)
        Areas.append(area / Big_S)
        if Big_S * 0.2 <= area <= Big_S * 0.3:
            Big_box = box.copy()
            # if show_process:
            #     leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            #     rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            #     topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            #     bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            #     cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            #     cv2.drawContours(origin_img, [box], 0, (255, 0, 0), 5)

            #     cv2.circle(origin_img, leftmost, 5, (0, 0, 255), -1)
            #     cv2.circle(origin_img, rightmost, 5, (0, 0, 255), -1)
            #     cv2.circle(origin_img, topmost, 5, (0, 0, 255), -1)
            #     cv2.circle(origin_img, bottommost, 5, (0, 0, 255), -1)

            #     hull = cv2.convexHull(cnt, returnPoints=False)
            #     defects = cv2.convexityDefects(cnt, hull)

            #     for i in range(defects.shape[0]):
            #         s, e, f, d = defects[i, 0]
            #         start = tuple(cnt[s][0])
            #         end = tuple(cnt[e][0])
            #         far = tuple(cnt[f][0])
            #         cv2.line(img, start, end, [0, 255, 0], 2)
            #         cv2.circle(img, far, 5, [0, 255, 255], -1)
            break
    # if show_process:
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(origin_img, cmap=plt.cm.gray)
    #     plt.show()
    return Big_box


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


def Count_nonzero_num(Big_box, temp_bin_img, show_process=False):
    # ------------- NEW WAY TO CAL SMALL BOXES
    Offset = (0, 93 / 386, 0.5, 293 / 386, 1)  # [0, 9 / 38, 0.5, 29 / 38, 1]
    Small_boxes = list()
    Vector_AB = [Big_box[1][0] - Big_box[0][0], Big_box[1][1] - Big_box[0][1]]
    Vector_DC = [Big_box[2][0] - Big_box[3][0], Big_box[2][1] - Big_box[3][1]]
    Main_Lines = list()
    for i in Offset:
        A_Point = [Big_box[0][0] + Vector_AB[0] * i, Big_box[0][1] + Vector_AB[1] * i]
        C_Point = [Big_box[3][0] + Vector_DC[0] * i, Big_box[3][1] + Vector_DC[1] * i]
        Main_Lines.append((A_Point, C_Point[0] - A_Point[0], C_Point[1] - A_Point[1]))
    for OffsetID in range(1, 5):
        for MainLineID in range(1, 5):
            Temp_arr = list()
            for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                Temp_arr.append(
                    [
                        round(
                            Main_Lines[MainLineID + i][0][0]
                            + Main_Lines[MainLineID + i][1] * Offset[OffsetID + j]
                        ),
                        round(
                            Main_Lines[MainLineID + i][0][1]
                            + Main_Lines[MainLineID + i][2] * Offset[OffsetID + j]
                        ),
                    ]
                )
            Small_boxes.append(np.array(Temp_arr))
    # print(Small_boxes)
    for box in Small_boxes:
        cv2.drawContours(temp_bin_img, [box], 0, 0, 2)
    # if show_process:
    #     print("Adjust img--------------")
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    #     plt.show()
    return np.count_nonzero(temp_bin_img)


def Adjust_Big_square_coordinates(Big_box, bin_img, show_process=False):
    Stop = False
    Loop_time = 0
    Centre_of_Big_box = [
        sum([Big_box[i][0] for i in range(4)]) / 4,
        sum([Big_box[i][1] for i in range(4)]) / 4,
    ]
    Important_area = Big_box.copy()
    for i in range(4):
        Important_area[i][0] = (
            Centre_of_Big_box[0] + (Big_box[i][0] - Centre_of_Big_box[0]) * 1.02
        )
        Important_area[i][1] = (
            Centre_of_Big_box[1] + (Big_box[i][1] - Centre_of_Big_box[1]) * 1.02  # 0.95
        )

    maskImage = np.zeros(bin_img.shape, dtype=np.uint8)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(maskImage, cmap=plt.cm.gray)
    # plt.show()
    cv2.drawContours(maskImage, [Important_area], 0, 255, -1)
    bin_img = cv2.bitwise_and(bin_img, maskImage)

    plt.figure(figsize=(10, 10))
    plt.imshow(bin_img, cmap=plt.cm.gray)
    plt.show()

    temp_bin_img = bin_img.copy()
    NonZero_num = Count_nonzero_num(Big_box, temp_bin_img)
    # if show_process:
    #     print("====== Non Zero num: ", NonZero_num, "    =============== !")

    while not Stop:
        Best_way = [-1, -1, -1]
        for i in range(4):
            for offset_x in range(-10, 11, 1):  # -10,11,1  -11,12,2
                for offset_y in range(-10, 11, 1):  # -10,11,1
                    if offset_x == offset_y == 0:
                        continue
                    temp_bin_img = bin_img.copy()
                    Big_box[i][0] += offset_x
                    Big_box[i][1] += offset_y
                    temp_NonZero_num = Count_nonzero_num(
                        Big_box, temp_bin_img, show_process=False
                    )
                    if temp_NonZero_num < NonZero_num:
                        NonZero_num = temp_NonZero_num
                        Best_way = [offset_x, offset_y, i]
                        # Count_nonzero_num(Big_box, temp_bin_img, show_process=True)
                    Big_box[i][0] -= offset_x
                    Big_box[i][1] -= offset_y
        if Best_way != [-1, -1, -1]:

            Big_box[Best_way[2]][0] += Best_way[0]
            Big_box[Best_way[2]][1] += Best_way[1]
            # if show_process:
            #     print("Best way: ", Best_way)
            #     Count_nonzero_num(Big_box, temp_bin_img, show_process=True)
        else:
            break
        Loop_time += 1
        if Loop_time == 50:
            Stop = True
            if show_process:
                print("----    Stop by time out    ------------------")
    # if show_process:
    #     print("Best Ans:", NonZero_num, "Loop time", Loop_time, "   =============\n\n")
    return Big_box


def Collect_related_points(target_point, Small_boxes):
    Target_point_addresses = list()
    Related_points_addresses = list()
    for i in range(16):
        for j in range(4):
            if Small_boxes[i][j] == target_point:
                Target_point_addresses.append((i, j))
                related_j = j - 1
                if related_j < 0:
                    related_j = 3
                if Small_boxes[i][related_j] not in Related_points_addresses:
                    Related_points_addresses.append((i, related_j))
                related_j = j + 1
                if related_j > 3:
                    related_j = 0
                if Small_boxes[i][related_j] not in Related_points_addresses:
                    Related_points_addresses.append((i, related_j))
    return Target_point_addresses, Related_points_addresses


def Collect_points_in_same_coordinate(target_point, Small_boxes):
    Target_point_addresses = list()
    for i in range(16):
        for j in range(4):
            if Small_boxes[i][j] == target_point:
                Target_point_addresses.append((i, j))
    return Target_point_addresses


def Smallest_area(Points, Small_boxes):
    X = [Small_boxes[i[0]][i[1]][0] for i in Points]
    Y = [Small_boxes[i[0]][i[1]][1] for i in Points]
    # print(X)
    # print(Y)
    return ((min(X), min(Y)), (max(X), max(Y)))


def Count_0_125(
    Point,
    Related_points_addresses,
    Small_boxes,
    temp_bin_img,
    Left_top_point,
    show_process=False,
    Clear_obj=False,
):
    # plt.figure(figsize=(10, 10))
    # plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    # plt.show()
    if Clear_obj:
        temp_bin_img = Dilatation(temp_bin_img, 7, 1)
        # temp_bin_img = Erode(temp_bin_img, 5, 3)
    # temp_bin_img = Dilitation(temp_bin_img, 3, 5)
    # temp_bin_img = Erode(temp_bin_img, 3, 5)
    # if show_process:
    #     print("___Small area___Start")
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    #     plt.show()

    Point[0] -= Left_top_point[0]
    Point[1] -= Left_top_point[1]
    Const_point = np.array([0, 0])
    for i in Related_points_addresses:
        Const_point[0] = Small_boxes[i[0]][i[1]][0] - Left_top_point[0]
        Const_point[1] = Small_boxes[i[0]][i[1]][1] - Left_top_point[1]
        cv2.line(
            temp_bin_img, Point, Const_point, 125, 5
        )  #  <<<============                                         ========= X11  6 oke 2 4 very good
    # if show_process:
    #     print("___Small area___End")
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    #     plt.show()
    Zero_new = np.count_nonzero(temp_bin_img == 0)
    Line_new = np.count_nonzero(temp_bin_img == 125)
    return Zero_new, Line_new


def Adjust_point_coordinate(point, Small_boxes, bin_img, Clear_obj=False):
    rows, cols = bin_img.shape
    # print(rows, cols, "<----- row,col")
    Target_point_addresses, Related_points_addresses = Collect_related_points(
        point, Small_boxes
    )
    # print(Target_point_addresses)
    # print(Related_points_addresses)
    Left_top_point, Right_down_point = Smallest_area(
        Related_points_addresses + [Target_point_addresses[0]], Small_boxes
    )
    # print(
    #     [
    #         Left_top_point[0],
    #         Right_down_point[0] + 1,
    #         Left_top_point[1],
    #         Right_down_point[1] + 1,
    #     ]
    # )
    # ---- !!! --- Not run this
    # print("BEFORE")
    # temp_bin_img = bin_img[
    #     Left_top_point[1] : Right_down_point[1] + 1,
    #     Left_top_point[0] : Right_down_point[0] + 1,
    # ]  # ------???
    # plt.figure(figsize=(10, 10))
    # plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    # plt.show()
    # print("LEFT", Left_top_point)
    Left_top_point = (max(Left_top_point[0] - 30, 0), max(Left_top_point[1] - 30, 0))
    # print("LEFT", Left_top_point)
    # print("RIGHT", Right_down_point)
    Right_down_point = (
        min(Right_down_point[0] + 30, cols - 1),
        min(Right_down_point[1] + 30, rows - 1),
    )
    # print("RIGHT", Right_down_point)
    # print("AFTER")
    temp_bin_img = bin_img[
        Left_top_point[1] : Right_down_point[1] + 1,
        Left_top_point[0] : Right_down_point[0] + 1,
    ]  # ------???
    # plt.figure(figsize=(10, 10))
    # plt.imshow(temp_bin_img, cmap=plt.cm.gray)
    # plt.show()
    # print(temp_bin_img, "Temp bin img <====")

    # print(
    #     [
    #         Left_top_point[0],
    #         Right_down_point[0] + 1,
    #         Left_top_point[1],
    #         Right_down_point[1] + 1,
    #     ]
    # )
    Point = Small_boxes[Target_point_addresses[0][0]][
        Target_point_addresses[0][1]
    ].copy()
    # print("Pre Start adjust", Point, type(Point))
    Zero_origin, Line_origin = Count_0_125(
        Point.copy(),
        Related_points_addresses,
        Small_boxes,
        temp_bin_img.copy(),
        Left_top_point,
        Clear_obj=Clear_obj,
    )
    Best_Ans = [0, 0]
    if Clear_obj:
        Range = [-4, 5]
    else:
        Range = [-1, 2]
    # print("Start adjust", Point, type(Point))
    for i in range(Range[0], Range[1], 1):  # -7,8,1   -3,4,1 -1,2,1
        for j in range(Range[0], Range[1], 1):
            if i == j == 0:
                continue
            Point[0] += i
            Point[1] += j
            # print(Point, i, j, "<<----------")
            Zero_new, Line_new = Count_0_125(
                Point.copy(),
                Related_points_addresses,
                Small_boxes,
                temp_bin_img.copy(),
                Left_top_point,
                Clear_obj=Clear_obj,
            )
            # --- Ver 1 ---
            # if Zero_new - Zero_origin > 51:  # 31 7 near good
            #     Zero_origin = Zero_new
            #     Line_origin = Line_new
            #     Best_Ans = [i, j]
            # elif -1 <= Zero_new - Zero_origin <= 51 and Line_new < Line_origin:
            if 1 <= Zero_new - Zero_origin and 1 <= Line_origin - Line_new:
                Zero_origin = Zero_new
                Line_origin = Line_new
                Best_Ans = [i, j]
            # --- Ver 2 ---
            # if (abs(Zero_new - Zero_origin) < 3 and Line_new < Line_origin) or (
            #     Zero_origin - Zero_new >= 3
            # ):
            #     Zero_origin = Zero_new
            #     Line_origin = Line_new
            #     Best_Ans = [i, j]
            Point[0] -= i
            Point[1] -= j
    # print("===> Best ans small:", Best_Ans)
    for i in Target_point_addresses:
        Small_boxes[i[0]][i[1]][0] += Best_Ans[0]
        Small_boxes[i[0]][i[1]][1] += Best_Ans[1]
    return Small_boxes


def Adjust_Small_squares_coordinates(Small_boxes, bin_img, show_process=False):
    # print("Adjust_Small_squares_coordinates___________")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(bin_img, cmap=plt.cm.gray)
    # plt.show()

    # --- Ver 1 ---
    # Approved_points = set()
    # for i in range(16):
    #     for j in range(4):
    #         if tuple(Small_boxes[i][j]) not in Approved_points:
    #             Small_boxes = Adjust_point_coordinate(
    #                 Small_boxes[i][j], Small_boxes, bin_img
    #             )
    #             Approved_points.add(tuple(Small_boxes[i][j]))

    # --- Ver 2 ---
    Bad_points = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [3, 1],
        [7, 1],
        [11, 1],
        [15, 1],
        [15, 2],
        [14, 2],
        [13, 2],
        [12, 2],
        [12, 3],
        [12, 0],
        [8, 0],
        [4, 0],
    ]
    # + [
    #     [5, 0],
    #     [6, 0],
    #     [7, 0],
    #     [9, 0],
    #     [10, 0],
    #     [11, 0],
    #     [13, 0],
    #     [14, 0],
    #     [15, 0],
    # ] * 7
    # for i, j in Bad_points:
    #     # print(i, j, "<-----------???")
    #     Small_boxes = Adjust_point_coordinate(
    #         Small_boxes[i][j], Small_boxes, bin_img, Clear_obj=True
    #     )
    for i, j in Bad_points * 10000:
        # print(i, j, "<-----------???")
        Small_boxes = Adjust_point_coordinate(
            Small_boxes[i][j], Small_boxes, bin_img, Clear_obj=False
        )
    return Small_boxes


def Clear_Outside(bin_img, Big_box):
    Centre_of_Big_box = [
        sum([Big_box[i][0] for i in range(4)]) / 4,
        sum([Big_box[i][1] for i in range(4)]) / 4,
    ]
    Important_area = Big_box.copy()
    for i in range(4):
        Important_area[i][0] = (
            Centre_of_Big_box[0] + (Big_box[i][0] - Centre_of_Big_box[0]) * 1.2
        )
        Important_area[i][1] = (
            Centre_of_Big_box[1] + (Big_box[i][1] - Centre_of_Big_box[1]) * 1.2
        )
    maskImage = np.zeros_like(bin_img, dtype=np.uint8)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(maskImage, cmap=plt.cm.gray)
    # plt.show()
    cv2.drawContours(maskImage, [Important_area], -1, 255, -1)
    maskImage = ~maskImage
    # plt.figure(figsize=(10, 10))
    # plt.imshow(maskImage, cmap=plt.cm.gray)
    # plt.show()
    bin_img = cv2.bitwise_or(bin_img, maskImage)
    bin_img = Erode(bin_img, 3, 1)
    bin_img = Dilatation(bin_img, 3, 1)
    bin_img = Dilatation(bin_img, 7, 1)
    bin_img = Erode(bin_img, 7, 1)
    bin_img = Dilatation(bin_img, 3, 5)
    # bin_img = Erode(bin_img, 3, 2)

    # bin_img = Dilitation(bin_img, 3, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(bin_img, cmap=plt.cm.gray)
    plt.show()
    return bin_img


def Move_side_points_away(Small_boxes):
    CenterPoint = Small_boxes[10][0]
    Side_points = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [3, 1],
        [7, 1],
        [11, 1],
        [15, 1],
        [15, 2],
        [14, 2],
        [13, 2],
        [12, 2],
        [12, 3],
        [12, 0],
        [8, 0],
        [4, 0],
    ]
    for i in Side_points:
        Point = Small_boxes[i[0]][i[1]].copy()
        Points_in_same_coordinate = Collect_points_in_same_coordinate(
            Point, Small_boxes
        )
        Point[0] = Point[0] + (10 if (Point[0] - CenterPoint[0]) > 0 else -10)
        Point[1] = Point[1] + (10 if (Point[1] - CenterPoint[1]) > 0 else -10)
        for x, y in Points_in_same_coordinate:
            Small_boxes[x][y][0] = Point[0]
            Small_boxes[x][y][1] = Point[1]
    return Small_boxes


def Nex_move(Move, min_val, max_val, step):
    for i in range(25):
        if Move[i] != [max_val, max_val]:
            if Move[i][0] < max_val:
                Move[i][0] += step
            else:
                Move[i][1] += step
            for j in range(i):
                Move[j] = [min_val, min_val].copy()
            return Move
    return None


def Count_White_Black(Points, bin_img):
    for i in range(25):
        j = i + 1
        if j % 5 != 0:
            cv2.line(bin_img, Points[i], Points[j], 125, 2)
        j = i + 5
        if j < 25:
            cv2.line(bin_img, Points[i], Points[j], 125, 2)
    White = np.count_nonzero(bin_img == 255)
    Black = np.count_nonzero(bin_img == 0)
    return (White, Black)


def Adjust_all_points(Small_boxes, bin_img, show_process=False):
    Points = list()
    Points_id = (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (7, 1),
        (8, 0),
        (9, 0),
        (10, 0),
        (11, 0),
        (11, 1),
        (12, 0),
        (13, 0),
        (14, 0),
        (15, 0),
        (15, 1),
        (12, 3),
        (13, 3),
        (14, 3),
        (15, 3),
        (15, 2),
    )
    for i, j in Points_id:
        Points.append(Small_boxes[i][j].copy())
    Move = [[-5, -5] for _ in range(25)]
    All_Cases = list()
    Case = {"Move": "?", "White_points": 0, "Black_points": 0}
    Move_id = 0
    while Move is not None:
        Valid = True
        i = 0
        while i < 25:
            Points[i][0] += Move[i][0]
            Points[i][1] += Move[i][1]
            if bin_img[Points[i][0]][Points[i][1]] == 0:
                Valid = False
                i += 1
                break
            i += 1
        if Valid:
            Case["White_points"], Case["Black_points"] = Count_White_Black(
                Points, bin_img.copy()
            )
            Case["Move"] = [_.copy() for _ in Move]
            All_Cases.append(Case.copy())
        for j in range(i):
            Points[j][0] -= Move[j][0]
            Points[j][1] -= Move[j][1]
        # print(Move)
        Move = Nex_move(Move, -5, 5, 1)
        Move_id += 1
        if Move_id % 100 == 0:
            print(Move_id)
            print(Move)
            print(Points)
    Min_White_Cases = sorted(All_Cases, key=lambda x: x["White_points"])[:31]
    Best_Cases = sorted(Min_White_Cases, key=lambda x: x["Black_points"], reverse=True)[
        0
    ]
    for i in range(25):
        Points[i][0] += Best_Cases["Move"][i][0]
        Points[i][1] += Best_Cases["Move"][i][1]
    Left_top_id = 0
    for i in range(16):
        Small_boxes[i][0] = Points[Left_top_id]
        Small_boxes[i][1] = Points[Left_top_id + 1]
        Small_boxes[i][2] = Points[Left_top_id + 6]
        Small_boxes[i][3] = Points[Left_top_id + 5]
        if i % 4 == 3:
            Left_top_id += 2
        else:
            Left_top_id += 1
    return Small_boxes


def Process_with_path(win_path, show_process=False):
    img = Load_img(win_path)
    origin_img = Load_img_in_color(win_path)
    rows, cols = img.shape
    Threshold = Find_suitable_threshold_ver2(img)
    img = Binary_Threshold(img, Threshold)
    BigSquareImg = Dilatation(img, size=3, repeat=6)
    BigSquareImg = Erode(BigSquareImg, size=3, repeat=12)
    BigSquareImg = Dilatation(BigSquareImg, size=3, repeat=6)

    Big_box = Contour(
        BigSquareImg, show_process=show_process, origin_img=origin_img.copy()
    )
    Big_box = Adjust_Big_square_coordinates(Big_box, img, show_process=show_process)
    # img = Clear_Outside(img, Big_box)
    # ------------- NEW WAY TO CAL SMALL BOXES
    Offset = (0, 93 / 386, 0.5, 293 / 386, 1)  # [0, 9 / 38, 0.5, 29 / 38, 1]
    Small_boxes = list()
    Vector_AB = [Big_box[1][0] - Big_box[0][0], Big_box[1][1] - Big_box[0][1]]
    Vector_DC = [Big_box[2][0] - Big_box[3][0], Big_box[2][1] - Big_box[3][1]]
    Main_Lines = list()
    for i in Offset:
        A_Point = [Big_box[0][0] + Vector_AB[0] * i, Big_box[0][1] + Vector_AB[1] * i]
        C_Point = [Big_box[3][0] + Vector_DC[0] * i, Big_box[3][1] + Vector_DC[1] * i]
        Main_Lines.append((A_Point, C_Point[0] - A_Point[0], C_Point[1] - A_Point[1]))
    for OffsetID in range(1, 5):
        for MainLineID in range(1, 5):
            Temp_arr = list()
            for i, j in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                Temp_arr.append(
                    [
                        round(
                            Main_Lines[MainLineID + i][0][0]
                            + Main_Lines[MainLineID + i][1] * Offset[OffsetID + j]
                        ),
                        round(
                            Main_Lines[MainLineID + i][0][1]
                            + Main_Lines[MainLineID + i][2] * Offset[OffsetID + j]
                        ),
                    ]
                )
            Small_boxes.append(Temp_arr)
    # print(Small_boxes)
    # # Small_boxes = Move_side_points_away(Small_boxes)
    # print(Small_boxes)
    # Small_boxes = Adjust_Small_squares_coordinates(
    #     Small_boxes, img, show_process=show_process
    # )
    # Small_boxes = Adjust_all_points(Small_boxes, img)
    Small_boxes = [np.array(i) for i in Small_boxes]
    # print(Small_boxes, "<=======")
    if show_process:
        color = [(255, 0, 0)] + [(0, 0, 255), (0, 0, 255), (0, 0, 255)] * 6
        i = 0
        for box in Small_boxes:
            cv2.drawContours(origin_img, [box], 0, color[i], 2)
            i += 1
        plt.figure(figsize=(10, 10))
        plt.imshow(origin_img, cmap=plt.cm.gray)
        plt.show()
    if show_process:
        return origin_img
    return Big_box, Small_boxes


# ---------------------------------------------------------------------------------------------------------
def isInsideTriangle(A, B, C, P):
    # print("A:", A)
    # print("P:", P)
    # Calculate the barycentric coordinates
    # of point P with respect to triangle ABC
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    a = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / denominator
    b = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / denominator
    c = 1 - a - b

    # Check if all barycentric coordinates
    # are non-negative
    if a >= 0 and b >= 0 and c >= 0:
        return True
    else:
        return False


def Check_inside(Point, Box):
    return any(
        [
            isInsideTriangle(Box[0], Box[1], Box[2], Point),
            isInsideTriangle(Box[0], Box[2], Box[3], Point),
        ]
    )


def Cal_dist(A, B):
    return math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2))


def Check_in_line(A, B, C, threshold):
    if (Cal_dist(C, A) + Cal_dist(C, B)) / Cal_dist(A, B) < threshold:
        return True
    return False


def In_which_box(contours, Small_Boxes):
    Vote = [0] * 17
    for Point in contours:
        # print("Point: ", Point)
        for i in range(16):
            if Check_inside(Point[0], Small_Boxes[i]):
                Vote[i] += 1
                if i in (0, 1, 2, 3):
                    if Vote[i] > 11:
                        if Check_in_line(
                            Small_Boxes[i][0], Small_Boxes[i][1], Point[0], 1.1
                        ):
                            Vote[i] -= 5
                if i in (0, 4, 8, 12):
                    if Vote[i] > 11:
                        if Check_in_line(
                            Small_Boxes[i][3], Small_Boxes[i][0], Point[0], 1.1
                        ):
                            Vote[i] -= 5
                if i in (3, 7, 11, 15):
                    if Check_in_line(
                        Small_Boxes[i][1], Small_Boxes[i][2], Point[0], 1.1
                    ):
                        Vote[i] += 11
                if i in (12, 13, 14, 15):
                    if Check_in_line(
                        Small_Boxes[i][2], Small_Boxes[i][3], Point[0], 1.1
                    ):
                        Vote[i] += 11
                break
        else:
            Vote[16] += 1
    Max_vote = max(Vote)
    if Max_vote > 0:
        Index = Vote.index(Max_vote)
        if Index == 16:
            return None
        else:
            return Vote.index(Max_vote)
    return None


def Show_Contour(Origin_path, Mask_path):
    Mask_img_gray = Load_img(Mask_path)
    Origin_img = Load_img_in_color(Origin_path)
    contours, hierarchy = cv2.findContours(
        Mask_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        cv2.drawContours(Origin_img, [c], 0, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(Origin_img, cmap=plt.cm.gray)
    plt.show()


def Count_Yeast_in_16_Squares(Origin_path, Mask_path, show_process=False):
    Big_box, Small_boxes = Process_with_path(Origin_path)
    print(Small_boxes)
    # Mask_img = Load_img_in_color(Mask_path)
    Mask_img_gray = Load_img(Mask_path)
    Origin_img = Load_img_in_color(Origin_path)
    # rows, cols = Mask_img_gray.shape
    # Big_S = rows * cols
    Big_S = cv2.contourArea(Big_box)
    Min_S_limit = Big_S * 0.00015
    Max_S_limit = Big_S * 0.01
    if show_process:
        print("Max/Min S limit", Max_S_limit, Min_S_limit)
        for box in Small_boxes:
            cv2.drawContours(Origin_img, [box], 0, (94, 53, 177), 2)
    contours, hierarchy = cv2.findContours(
        Mask_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # print(sorted([cv2.contourArea(c) for c in contours]))
    Colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] * 6
    Count_yeast = [0] * 16
    for c in contours:
        # print(c)
        # calculate moments for each contour
        area = cv2.contourArea(c)
        # print(area)
        if area < Min_S_limit or area > Max_S_limit:
            continue
        # print(area)
        M = cv2.moments(c)
        # if M["m00"] == 0:
        #     continue
        In_box = In_which_box(c, Small_boxes)
        if In_box != None:
            Count_yeast[In_box] += 1
            # calculate x,y coordinate of center
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # cv2.circle(Origin_img, (cX, cY), 3, Colors[In_box], -1)
            cv2.drawContours(Origin_img, [c], 0, Colors[In_box], 2)
    if show_process:
        Color_for_direction = [
            (255, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 0),
        ]
        Direction = [(3, 0), (0, 1), (1, 2), (2, 3)]
        for i in range(4):
            cv2.line(
                Origin_img,
                Small_boxes[0][Direction[i][0]],
                Small_boxes[0][Direction[i][1]],
                Color_for_direction[i],
                2,
            )
        Colors = [(0, 0, 125), (125, 0, 0), (0, 125, 0)] * 6
        for index in range(16):
            cv2.putText(
                Origin_img,
                "{}-{}".format(index + 1, Count_yeast[index]),
                (
                    sum(i[0] for i in Small_boxes[index]) // 4 - 25,
                    sum(i[1] for i in Small_boxes[index]) // 4,
                ),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1,
                Colors[index],
                4,
            )
        plt.figure(figsize=(10, 10))
        plt.imshow(Origin_img, cmap=plt.cm.gray)
        plt.show()
        return Origin_img
    else:
        return Count_yeast
