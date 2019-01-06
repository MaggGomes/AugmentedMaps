import cv2
import numpy as np
import math
import yaml

from typing import Tuple, List
from PyQt5 import QtGui as gui

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)


# Draws center circle in the map
def draw_center_map(img, width, height):
    cv2.circle(img, (int(width/2), int(height/2)),
               6, (0), 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (int(width/2), int(height/2)),
               6, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    return img


# Return nearest Point of Interest
def get_nearest_interestpoint(image_prepared, matrix, w, h):
    nearest_interestpoint = [None, None, None, None]

    interestpointF = None
    for interestpoint in image_prepared.interestPoints:
        print(
            f"Points of Interest - Coord Xi: {interestpoint.x}")
        print(
            f"Points of Interest - Coord Yi: {interestpoint.y}")
        print(
            f"Points of Interest - Width: {interestpoint.w}")
        print(
            f"Points of Interest - Height: {interestpoint.h}")

        # Gets the corners of the interestpoint
        pts_interestpoint = np.float32([[interestpoint.x, interestpoint.y], [interestpoint.x, interestpoint.y + interestpoint.h - 1], [interestpoint.x +
                                                                                                                                       interestpoint.w - 1, interestpoint.y + interestpoint.h - 1], [interestpoint.x + interestpoint.w - 1, interestpoint.y]]).reshape(-1, 1, 2)

        # Project corners into frame
        dst_interestpoint = cv2.perspectiveTransform(
            pts_interestpoint, matrix)

        # Calculares centroid of the transformed interestpoint
        centroid_interestpoint = get_centroid(
            (dst_interestpoint[0][0], dst_interestpoint[1][0], dst_interestpoint[2][0], dst_interestpoint[3][0]))

        # Calculates distance between the
        distance = distante_between_points(
            centroid_interestpoint, (w/2, h/2))

        print(distance)

        if nearest_interestpoint[1] is None or distance < nearest_interestpoint[1]:
            print("Changed nearest Point of Interest")
            nearest_interestpoint = [
                dst_interestpoint, distance, interestpoint.name, interestpoint.images[0]]
            interestpointF = interestpoint

    return interestpointF.x, interestpointF.y, nearest_interestpoint


# Returns the compass points
def get_compass_points(width, height):
    pts = np.float32(
        [[int(width/2) + 30, int(height/2)], [int(width/2) + 40, int(height/2) + 30], [int(width/2) + 50, int(height/2)], [int(width/2) + 40, int(height/2) - 30]]).reshape(-1, 1, 2)

    return pts


# Returns the header image points
def get_header_points(xi, yi, w):
    pts = np.float32(
        [[xi, yi - 30], [xi, yi], [w, yi], [w, yi - 30]]).reshape(-1, 1, 2)

    return pts


# Get descriptors from an image
def get_features(img) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(img, None)


# Match descriptors between 2 images
def match_descriptors(src_des: np.ndarray, target_des: np.ndarray):
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(src_des, target_des, k=2)

    # Ratio test as per Lowe's paper
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


# Histogram equalization
def histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist_eq = clahe.apply(img_grayscale)
    return img_hist_eq


# Converts keypoints to a dictionary to allow its serialization
def keypoints_to_kpdict(kps):
    keypoints = []
    for keypoint in kps:
        temp: dict = {
            'pt': keypoint.pt,
            'angle': keypoint.angle,
            'response': keypoint.response,
            'octave': keypoint.octave,
            'class_id': keypoint.class_id
        }

        keypoints.append(temp)

    return keypoints


# Calculates centroid of the polygon
def get_centroid(vertexes):
    print("Calculating the centroid of a polygon representing a point of interest")
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return(_x, _y)


# Calculates distance between 2 points
def distante_between_points(p1, p2):
    print("Calculating the distance between a point of interest and the center")
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance


# Converts qimage to n umpy
def qimage_to_numpy(image: gui.QImage):
    ptr = image.bits()
    w, h, _ = image.width(), image.height(), image.depth()
    ptr.setsize(w * h * 4)
    return np.array(ptr).reshape(h, w, 4)


# Converts numpy to qimage
def numpy_to_qimage(src: np.array):
    shape = src.shape
    h, w = shape[0], shape[1]
    d = 1
    if len(shape) == 3:
        d = shape[2]
    return gui.QImage(src, w, h, w * d, gui.QImage.Format_RGB888 if d != 1 else gui.QImage.Format_Grayscale8)


# Converts an image to qimage
def image_to_qimage(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return numpy_to_qimage(img_rgb)


def camera_calibration_matrix():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(0)
    found = 0
    while(found < 30):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.imwrite('photo.png' , img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            found += 1

        cv2.imshow('img', img)
        cv2.waitKey(10)
        if found == 30:
            cv2.imwrite ('output.png', img)
    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape [::-1], None, None)

    data = {'camera_matrix': np.asanyarray(mtx), 'dist_coeff': np.asarray(dist)}

    with open("camera_parameters.yaml", "w") as f:
        yaml.dump(data, f)


def projection_matrix(camera_parameters, homography):

    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(img, projection, base_pointsS, interestPointCentroid ):


    scale_matrix = np.eye(3) * 12
    base_points = []
    for p in base_pointsS:
        base_points.append(p)


    centroid_point = get_centroid((base_pointsS[0][0][0], base_pointsS[0][1][0], base_pointsS[0][2][0], base_pointsS[0][3][0]))
    centroid_point = interestPointCentroid
    f1 = [base_points[0][1][0][0], base_points[0][1][0][1], 0], [centroid_point[0], centroid_point[1], 60], [base_points[0][0][0][0], base_points[0][0][0][1], 0]
    f2 = [base_points[0][2][0][0], base_points[0][2][0][1], 0], [centroid_point[0], centroid_point[1], 60], [base_points[0][2][0][0], base_points[0][2][0][1], 0]
    f3 = [base_points[0][2][0][0], base_points[0][2][0][1], 0], [centroid_point[0], centroid_point[1], 60], [base_points[0][3][0][0], base_points[0][3][0][1], 0]
    f4 = [base_points[0][3][0][0], base_points[0][3][0][1], 0], [centroid_point[0], centroid_point[1], 60], [base_points[0][0][0][0], base_points[0][0][0][1], 0]
    faces = [f1, f2, f3, f4]

    faces =  np.float32([ [[-3,-3,0], [0,0,20], [3,-3,0]],  [[-3,3,0], [0,0,20], [-3,-3,0]], [[3,3,0], [0,0,20], [-3,3,0]], [[3,-3,0], [0,0,20], [3,3,0]]])


    for face in faces:
        points = np.dot(face, scale_matrix)
        w = centroid_point[0]
        h = centroid_point[1]
        points = np.array([[p[0] + w + 40, p[1] + h + 30, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        #dst = np.array([[p[0] + w, p[1] + h] for p[0] in dst])
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, (137, 27, 211))



    return img