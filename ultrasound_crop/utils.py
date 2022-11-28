import cv2 as cv
import numpy as np


def save_and_show(path, img, caption):  # save image to path
    cv.imwrite(path, img)
    cv.imshow(caption, img)


def maxAreaContour(contours):  # find a contour with max area
    maxArea = 0
    maxAreaIdx = 0
    PointsVector = []
    for c in range(len(contours)):
        points = cv.convexHull(contours[c])
        PointsVector.append(points)
        area = cv.contourArea(points)
        if area > maxArea:
            maxArea = area
            maxAreaIdx = c
    Points = PointsVector[maxAreaIdx]
    return Points


def drawConvexHull(img, pointsVector):  # draw convex hull in img
    total = len(pointsVector)
    for i in range(total):  # draw the convex hull
        x1, y1 = pointsVector[i % total][0]
        x2, y2 = pointsVector[(i + 1) % total][0]
        cv.circle(img, (x1, y1), 4, (255, 0, 0), 2, 8, 0)  # circle each vertex
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)  # draw the line
    save_and_show('./maxConvexHull.png', img, "max convex hull")


def cutConvexHull(img, convexHullPoints):  # cut ROI via convexHull's Points
    x_vector = convexHullPoints[:, :, 0]
    y_vector = convexHullPoints[:, :, 1]
    x1 = np.min(x_vector)
    y1 = np.min(y_vector)
    x2 = np.max(x_vector)
    y2 = np.max(y_vector)
    if x1 < 0:
       x1 = 0
    if x2 > img.shape[1]:
       x2 = img.shape[1]
    if y1 < 0:
        y1 = 0
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask_hull = cv.fillPoly(mask, [convexHullPoints], (255, 255, 255))
    ROI = cv.bitwise_and(mask_hull, img)
    img_cutted = ROI[y1: y2, x1: x2]
    return img_cutted


def extract_roi(img_path, save_path):  # cut ROI from an image, and save to path
    # src_img = cv.imread(img_path)
    src_img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)  # convert source image to gray

    dst_img = cv.Laplacian(gray_img, cv.CV_16S, ksize=3)
    Laplacian = cv.convertScaleAbs(dst_img)
    Laplacian[Laplacian == 255] = 0  # eliminate the word in figure

    ret, binary = cv.threshold(Laplacian, 0, 255, cv.THRESH_BINARY)  # convert to binary

    element = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))  # Opening
    binary_opening = cv.morphologyEx(binary, cv.MORPH_OPEN, element)

    contours, hierarchy = cv.findContours(binary_opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        cv.imencode('.bmp', src_img)[1].tofile(save_path)
        return
    maxAreaPoints = maxAreaContour(contours)  # Points of the max_area_contour
    # drawConvexHull(src_img, maxAreaPoints)  # draw the convex hull with max area

    image_cut = cutConvexHull(src_img, maxAreaPoints)
    # cv.imwrite(save_path, image_cut)
    cv.imencode('.bmp', image_cut)[1].tofile(save_path)


if __name__ == '__main__':  # test a single image
    src_img = cv.imread("C:\\Users\\Lenovo\\Desktop\\test.bmp")  # source image
    cv.imshow("source image", src_img)

    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)  # convert source image to gray
    print(gray_img.shape)
    dst_img = cv.Laplacian(gray_img, cv.CV_16S, ksize=3)
    Laplacian = cv.convertScaleAbs(dst_img)
    Laplacian[Laplacian == 255] = 0
    save_and_show('./laplacian.png', Laplacian, "laplacian")

    ret, binary = cv.threshold(Laplacian, 0, 255, cv.THRESH_BINARY)  # convert to binary
    save_and_show('./binary.png', binary, "binary")

    element = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))  # Opening
    binary_opening = cv.morphologyEx(binary, cv.MORPH_OPEN, element)
    save_and_show('./binary_opening.png', binary_opening, "binary open")

    contours, hierarchy = cv.findContours(binary_opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxAreaPoints = maxAreaContour(contours)  # Points of the max_area_contour
    drawConvexHull(src_img, maxAreaPoints)  # draw the convex hull with max area

    image_cut = cutConvexHull(src_img, maxAreaPoints)
    save_and_show('./image_cutted.png', image_cut, "image ROI")

    cv.waitKey()
