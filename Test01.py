from matplotlib import pyplot as plt
import Bai_toan_buong_dem_ver2
import time
import cv2

Large_input_path = (
    [
        "08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B1/Ethanol red_B1_19h_{}.jpg".format(
            i
        )
        for i in range(1, 37)
    ]
    + [
        "08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B2/Ethanol_red_B2_17h_{}.jpg".format(
            i
        )
        for i in range(1, 59)
    ]
    + [
        "08_Datasets/2025_02_19_Ethanol_red/Ethanol_red_B3_12h/Ethanol red_B3_12h_{}.jpg".format(
            i
        )
        for i in range(1, 72)
    ]
    + [
        "08_Datasets/2025_02_19_Ethanol_red/Ethanol_red_B3_19h/Ethanol red_B3_19h_{}.jpg".format(
            i
        )
        for i in range(1, 79)
    ]
)

Origin_mask_img_path = [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_{}_Ethanol_red/Ethanol_red_{}/Ethanol red_B{}_{}h_{}.jpg".format(
        i[0], i[1], i[2], i[3], i[4]
    )
    for i in [
        (18, "B1", 1, 19, 1),
        (18, "B1", 1, 19, 2),
        (19, "B3_12h", 3, 12, 1),
        (19, "B3_12h", 3, 12, 2),
        (19, "B3_19h", 3, 19, 1),
        (19, "B3_19h", 3, 19, 2),
    ]
] + [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B2/Ethanol_red_B{}_{}h_{}.jpg".format(
        i[0], i[1], i[2]
    )
    for i in [(2, 17, 1), (2, 17, 2)]
]
Mask_img_path = [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Mask/Mask{}.png".format(
        i
    )
    for i in range(8)
]
## ---- Corner coordinate of square only ----

index = 0
# Set to True for showing image, else False
show_process = True
# 41 52 21 26 36 37 38 39 40 42 43 45 46 47 49 50 51 53 54 55 56 57 58 59 61 62 63 64 65 66 67 68 69 71
for i in Large_input_path:
    # if index in [
    #     41,
    #     52,
    #     21,
    #     26,
    #     36,
    #     37,
    #     38,
    #     39,
    #     40,
    #     42,
    #     43,
    #     45,
    #     46,
    #     47,
    #     49,
    #     50,
    #     51,
    #     53,
    #     54,
    #     55,
    #     56,
    #     57,
    #     58,
    #     59,
    #     61,
    #     62,
    #     63,
    #     64,
    #     65,
    #     66,
    #     67,
    #     68,
    #     69,
    #     71,
    # ] and index in [38, 42, 62, 41]:
    if True:  # 57
        print(index)
        print(i)
        start_time = time.time()
        if show_process:
            Ans = Bai_toan_buong_dem_ver2.Process_with_path(
                i, show_process=show_process
            )
            # Bai_toan_buong_dem_ver2.Process_with_path_ver2(i, show_process=show_process)
            cv2.imwrite(
                "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Ans_ver7/Ans{}.png".format(
                    index
                ),
                Ans,
            )
        elif not show_process:
            Ans = Bai_toan_buong_dem_ver2.Process_with_path(
                i, show_process=show_process
            )
            print(Ans)
        end_time = time.time()
        print("Total running time: ", end_time - start_time, " ==================")
    index += 1

# ## ---- Count yeast in small square ----
# show_process = True
# for index in range(8):
#     print(index)
#     start_time = time.time()
#     if show_process:
#         print(Origin_mask_img_path[index])
#         print(Mask_img_path[index])
#         Ans = Bai_toan_buong_dem.Count_Yeast_in_16_Squares(
#             Origin_mask_img_path[index], Mask_img_path[index], show_process
#         )
#         Bai_toan_buong_dem.Show_Contour(
#             Origin_mask_img_path[index], Mask_img_path[index]
#         )
#         # cv2.imwrite(
#         #     "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/Mask/Ans{}.png".format(
#         #         index
#         #     ),
#         #     Ans,
#         # )
#     else:
#         Ans = Bai_toan_buong_dem.Count_Yeast_in_16_Squares(
#             Origin_mask_img_path[index], Mask_img_path[index], show_process
#         )
#         print(Ans)
#     end_time = time.time()
#     print("Total running time: ", end_time - start_time, " ==================")
