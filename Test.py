import Image_merge
import Corner_Coordinates
import cv2
from matplotlib import pyplot as plt
import time

Origin_img_path = [
    (
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\Para-40x-{}-1.jpg".format(
            i
        ),
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\Para-40x-{}-2.jpg".format(
            i
        ),
    )
    for i in [2, 5, 6, 7, 8, 9]
] + [
    (
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\T8_10x_3d_{}-1.jpg".format(
            i
        ),
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\T8_10x_3d_{}-2.jpg".format(
            i
        ),
    )
    for i in list(range(60, 65)) + list(range(66, 71)) + list(range(72, 83))
]
Mask_img_path = [
    (
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\Para-40x-{}-1.png".format(
            i
        ),
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\Para-40x-{}-2.png".format(
            i
        ),
    )
    for i in [2, 5, 6, 7, 8, 9]
] + [
    (
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\T8_10x_3d_{}-1.png".format(
            i
        ),
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Crop_and_Mask\T8_10x_3d_{}-2.png".format(
            i
        ),
    )
    for i in list(range(60, 65)) + list(range(66, 71)) + list(range(72, 83))
]

Origin_bug_path = None
Mask_bug_path = None

Large_input_path = (
    [
        "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B1/Ethanol red_B1_19h_{}.jpg".format(
            i
        )
        for i in range(1, 37)
    ]
    + [
        "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B2/Ethanol_red_B2_17h_{}.jpg".format(
            i
        )
        for i in range(1, 59)
    ]
    + [
        "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_19_Ethanol_red/Ethanol_red_B3_12h/Ethanol red_B3_12h_{}.jpg".format(
            i
        )
        for i in range(1, 72)
    ]
    + [
        "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_19_Ethanol_red/Ethanol_red_B3_19h/Ethanol red_B3_19h_{}.jpg".format(
            i
        )
        for i in range(1, 79)
    ]
)

Final_test_path = [
    (
        "D:/CODE library/OPENCV/Corner Coordinates of square/Cam2k_Buong_Dem/Data_test/Origin{}.png".format(
            i
        ),
        "D:/CODE library/OPENCV/Corner Coordinates of square/Cam2k_Buong_Dem/Data_test/Mask{}.png".format(
            i
        ),
    )
    for i in range(27)
]

index = 0
"""
for i in Final_test_path:
    print(i[0])
    Ans_img = Corner_Coordinates.Count_Yeast_in_16_Squares(i[0], i[1])
    cv2.imwrite(
        "D:\CODE library\OPENCV\Corner Coordinates of square\Cam2k_Buong_Dem\Ans_img\Ans{}.png".format(
            index
        ),
        Ans_img,
    )
    index += 1
"""

for i in Large_input_path:
    print(index)
    print(i)
    start_time = time.time()
    if True:
        Ans = Corner_Coordinates.Process_with_path(i)
        # cv2.imwrite(
        #     "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Ans_ver4/Ans{}.png".format(
        #         index
        #     ),
        #     Ans,
        # )
    end_time = time.time()
    print("Total running time: ", end_time - start_time, " ==================")
    index += 1


"""
for i in range(len(Origin_img_path)):
    print(Origin_img_path[i][0])
    Input_img = Image_merge.Merge_img(Origin_img_path[i][0], Origin_img_path[i][1])
    cv2.imwrite(
        "D:/CODE library/OPENCV/Corner Coordinates of square/Cam2k_Buong_Dem/Data_test/Origin{}.png".format(
            index
        ),
        Input_img,
    )
    Mask_img = Image_merge.Merge_img(Mask_img_path[i][0], Mask_img_path[i][1])
    cv2.imwrite(
        "D:/CODE library/OPENCV/Corner Coordinates of square/Cam2k_Buong_Dem/Data_test/Mask{}.png".format(
            index
        ),
        Mask_img,
    )
    Small_boxes = Corner_Coordinates.Process_with_merged_img(Input_img)
    color = [(255, 0, 255), (0, 255, 255), (0, 0, 255)] * 6
    color_i = 0
    for box in Small_boxes:
        cv2.drawContours(Mask_img, [box], 0, (0, 0, 0), 2)
        color_i += 1
    plt.figure(figsize=(10, 10))
    plt.imshow(Mask_img, cmap=plt.cm.gray)
    plt.show()
    index += 1
"""
