import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# images = glob.glob(
#     "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B1/Ethanol red_B1_19h_1.jpg"
# )
images = [
    "D:/CODE library/OPENCV/Corner Coordinates of square/08_Datasets/2025_02_18_Ethanol_red/Ethanol_red_B1/Ethanol red_B1_19h_1.jpg"
]
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (10, 10), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (10, 10), corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        print("Done=----")

cv2.destroyAllWindows()
