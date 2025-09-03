import Neubauer_chamber_RF_DETR
import Neubauer_chamber_RF_DETR_ver3_old
import Neubauer_chamber_RF_DETR_ver03
import Scale_image

import glob
from matplotlib import pyplot as plt
from rfdetr import RFDETRBase
import cv2
import time

# ------------------------------- RF-DETR model -----------------------------------------
# model = RFDETRBase(pretrain_weights="checkpoint_best_regular.pth")
# model = RFDETRBase(pretrain_weights="checkpoint_best_ema_ver02.pth")
model = RFDETRBase(pretrain_weights="checkpoint_best_ema_ver05.pth")

# origin_paths = list(glob.glob("Final_Ans/Origin_img_checked/*.jpg"))
imgsPath = "D://CODE library//OPENCV//Corner_Coordinates_of_square//Test_img"
origin_paths = list(glob.glob(f"{imgsPath}/*.jpg")) + list(
    glob.glob(f"{imgsPath}/*.jpeg")
)
imageErrorList = list()
countError = 0

for i in origin_paths:
    Img_name = (i.split(".")[0]).split("\\")[-1]

    print(Img_name)

    # mask_path = "Final_Ans/Mask_img_checked/{}_mask_inverted.png".format(Ans_name)

    start_time = time.time()

    # Output = RF_DETR_model.Find25Points(model, i, False)
    Output = Neubauer_chamber_RF_DETR_ver03.Find25Points(model, i, True)

    # Output = Neubauer_chamber_RF_DETR_ver3.Count_Yeast_in_16_Squares(
    #     model, i, mask_path, True, True
    # )

    end_time = time.time()
    print("Running time: ", end_time - start_time)

    plt.figure(figsize=(10, 10))
    plt.imshow(
        Output["AnnotatedImg"],
        cmap=plt.cm.gray,
    )
    plt.show()
    cv2.imwrite(
        "Final_Ans/Model_result_ver7/{}_Annotated.jpg".format(Img_name),
        cv2.cvtColor(Output["AnnotatedImg"], cv2.COLOR_RGB2BGR),
    )

    if Output["Error"] == None:
        print(Output["Points"])
        print("nanometer/pixel", Output["nanometer/pixel"])

        # print(Output["Count_yeast"])

        # plt.figure(figsize=(10, 10))
        # plt.imshow(
        #     Output["CountYeastImg"],
        #     cmap=plt.cm.gray,
        # )
        # plt.show()

        # plt.figure(figsize=(10, 10))
        # plt.imshow(
        #     Output["ScaledImg"],
        #     cmap=plt.cm.gray,
        # )
        # plt.show()

        cv2.imwrite(
            "Final_Ans/Model_result_ver7/{}_Scaled.jpg".format(Img_name),
            cv2.cvtColor(Output["ScaledImg"], cv2.COLOR_RGB2BGR),
        )

        # cv2.imwrite(
        #     "Final_Ans/Model_result_ver4/{}_result.jpg".format(Ans_name),
        #     Output["CountYeastImg"],
        # )

        Lines = list()
        for i in range(25):
            if i < 20:
                Lines.append((i, i + 5))
            if i % 5 != 4:
                Lines.append((i, i + 1))
        for i in Lines:
            cv2.line(
                Output["ScaledImg"],
                Output["Points"][i[0]],
                Output["Points"][i[1]],
                (255, 98, 41),  # BGR
                3,
            )
        plt.figure(figsize=(10, 10))
        plt.imshow(
            Output["ScaledImg"],
            cmap=plt.cm.gray,
        )
        plt.show()
    else:
        imageErrorList.append(Img_name)
        print(Output["Error"], "\n||||||||||||||||||" * 3)
        countError += 1

print("\n--> Summary: ", countError, "errors")
print(imageErrorList)
# ---------------------------- SCALE IMG --------------------------------
# Multi_resolution_img_path = list(
#     glob.glob("NgocAnh_work/2025_04_28_Buong_dem_cam_smartphone/Multi_resolution/*")
# )
# for i in Multi_resolution_img_path:
#     Ans_name = (i.split(".")[0]).split("\\")[-1]
#     img = Scale_image.Scale_image(i)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(
#         img,
#         cmap=plt.cm.gray,
#     )
#     plt.show()
#     cv2.imwrite(
#         "Scaled_image/{}_scaled.jpg".format(Ans_name),
#         img,
#     )


"IMG_7250_scaled -> found 2 big square,Error04,Error01"
