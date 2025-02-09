import sys
# 导入opencv、魔方识别与解算相关库
import cv2
import numpy as np
import math
import os
from timeit import default_timer as timer
import time
from sklearn import svm
import joblib
from copy import deepcopy
from helpers import ciede2000, bgr2lab
import kociemba
from constants import CUBE_PALETTE, COLOR_PLACEHOLDER
import matplotlib.pyplot as plt
# 导入串口操作类
import serial
import serial.tools.list_ports
# 导入线程类
import threading
# from threading import Thread
# ui设计导入库
from PyQt5 import uic
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QLabel, QTextBrowser)
# # from PyQt5.QtUiTools import QUiLoader
# from PyQt5.QtCore import QFile, pyqtSignal, QObject, Qt, QTimer, QDateTime, QPoint, QSize, QRect
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# 主要用于获取摄像头ID号，git克隆地址：git clone https://gitee.com/jiangbin2020/py-camera-list-project.git
# 参考文章：https://blog.csdn.net/babybin/article/details/122044565
from PyCameraList.camera_device import list_video_devices, list_audio_devices
# # 当前python版本
# print("system version:", sys.version)
# # 相机编号和名称
# cameras = list_video_devices()
# print("camera list:", dict(cameras))
# # 麦克风编号和名称
# audios = list_audio_devices()
# print("audio list:", dict(audios))


result_state = {  # URFDLB
    'U': [],
    'R': [],
    'F': [],
    'D': [],
    'L': [],
    'B': []
}  # 全局变量，注意：使用互斥锁

group_index = {
    'white': 0,
    'red': 0,
    'green': 0,
    'yellow': 0,
    'orange': 0,
    'blue': 0
}

lock = threading.RLock()  # 创建可重入锁


class mythread(threading.Thread):
    def __init__(self, camera_id, detection_mode, sides_position):
        super(mythread, self).__init__()
        self.camera_id = camera_id
        self.detection_mode = detection_mode
        self.sides_position = sides_position

        self.prominent_color_palette = {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255)
        }
        # Load colors from config and convert the list -> tuple.
        # self.cube_color_palette = config.get_setting(
        #     CUBE_PALETTE,
        #     self.prominent_color_palette
        # )
        # self.result_state = {}      # [center_color_name][9个色块的颜色RGB]
        # self.result_state = {}  # [中心方块名][9个色块的颜色名]
        global result_state  # 全局变量
        self.cube_color_palette = self.prominent_color_palette

        self.preview_colors_name = {}
        self.average_sticker_colors_name = {}  # 魔方色块颜色RGB名字存储

        self.average_sticker_colors = {}  # 魔方色块颜色RGB存储
        self.preview_state = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                              (255, 255, 255), (255, 255, 255), (255, 255, 255),
                              (255, 255, 255), (255, 255, 255), (255, 255, 255)]  # 可更新

    def get_dominant_color(self, roi):
        """
        Get dominant color from a certain region of interest.
        从某个感兴趣的区域获得主色调。
        :param roi: The image list.
        :returns: tuple
        """
        pixels = np.float32(roi.reshape(-1, 3))
        try:
            n_colors = 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts)]
            return tuple(dominant)
        except:
            return tuple((1, 1, 1))

    def get_closest_color(self, bgr):
        """
        Get the closest color of a BGR color using CIEDE2000 distance.
        使用CIEDE2000距离获得BGR颜色中最接近的颜色。
        :param bgr tuple: The BGR color to use.
        :returns: dict
        """
        lab = bgr2lab(bgr)
        distances = []
        for color_name, color_bgr in self.cube_color_palette.items():
            distances.append({
                'color_name': color_name,
                'color_bgr': color_bgr,
                'distance': ciede2000(lab, bgr2lab(color_bgr))
            })
        closest = min(distances, key=lambda item: item['distance'])
        return closest

    def draw_contours(self, Image, contours):
        for index, (x, y, w, h) in enumerate(contours):
            cv2.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def show_cubecolors(self, Image, colors, contours):
        for index, (x, y, w, h) in enumerate(contours):
            center_x = x + w / 2
            center_y = y + h / 2
            cv2.putText(Image, str(colors[index]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def update_preview_state(self, frame, contours):
        """
            Get the average color value for the contour for every X amount of frames
            to prevent flickering and more precise results.
    		获取每X帧的轮廓的平均颜色值，以防止闪烁和更精确的结果。
            """
        max_average_rounds = 8
        for index, (x, y, w, h) in enumerate(contours):
            if index in self.average_sticker_colors and len(
                    self.average_sticker_colors[index]) == max_average_rounds:
                sorted_items = {}
                for bgr in self.average_sticker_colors[index]:
                    key = str(bgr)
                    if key in sorted_items:
                        sorted_items[key] += 1
                    else:
                        sorted_items[key] = 1
                most_common_color = max(sorted_items, key=lambda i: sorted_items[i])
                self.average_sticker_colors[index] = []
                self.preview_state[index] = eval(most_common_color)
                break
            # ratio = 1
            roi = frame[y + 7:y + h - 7, x + 14:x + w - 14]
            # roi = frame[y + 7*ratio:y + h - 7*ratio, x + 14*ratio:x + w - 14*ratio]
            avg_bgr = self.get_dominant_color(roi)
            # closest_color = self.get_closest_color(avg_bgr)['color_bgr']
            closest_color = self.get_closest_color(avg_bgr)
            self.preview_state[index] = closest_color
            if index in self.average_sticker_colors:
                self.average_sticker_colors[index].append(closest_color['color_bgr'])
                self.average_sticker_colors_name[index].append([closest_color['color_name']])

                self.preview_colors_name[index] = closest_color['color_name']  # 当前面颜色
            else:
                self.average_sticker_colors[index] = [closest_color['color_bgr']]
                self.average_sticker_colors_name[index] = [closest_color['color_name']]

                self.preview_colors_name[index] = closest_color['color_name']

    def cube_sort(self, final_contours):
        # Sort contours on the y-value first.
        y_sorted = sorted(final_contours, key=lambda item: item[1])
        # Split into 3 rows and sort each row on the x-value.
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours

    def find_cube_neighbors(self, final_contours, radius):
        contour_neighbors = {}  # 相邻轮廓
        for index, contour in enumerate(final_contours):  # enumerate:当你需要在循环中获取可迭代对象的每个元素及其"索引"时，将经常用到该函数
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2  # 得到final_contours中每个轮廓的最小正矩形中心(x,y)坐标
            center_y = y + h / 2
            # radius = 1                                    # 半径(根据实际情况选择)
            # 为当前轮廓创建9个位置，它们是相邻的。我们将使用它来检查每个轮廓有多少相邻轮廓。所有这些都匹配
            # 的唯一方法是当前轮廓是魔方的中心。如果我们找到了中心，我们也知道了所有的邻边，从而知道
            # 了所有的轮廓，从而知道这个形状可以被认为是一个3x3x3的立方体。当我们找到这些轮廓时，我们将
            # 它们分类并返回。
            # [ (x-w*r,y-h*r),    (x,y-h*r),     (x+w*r,y-h*r)
            #  (x-w*r,y)    ,    (x,y)    ,     (x+w*r,y)
            #  (x-w*r,y+h*r),    (x,y+h*r),     (x+w*r,y+h*r) ]
            neighbor_positions = [
                # top left
                [(center_x - w * radius), (center_y - h * radius)],
                # top middle
                [center_x, (center_y - h * radius)],
                # top right
                [(center_x + w * radius), (center_y - h * radius)],
                # middle left
                [(center_x - w * radius), center_y],
                # center
                [center_x, center_y],
                # middle right
                [(center_x + w * radius), center_y],
                # bottom left
                [(center_x - w * radius), (center_y + h * radius)],
                # bottom middle
                [center_x, (center_y + h * radius)],
                # bottom right
                [(center_x + w * radius), (center_y + h * radius)],
            ]
            for neighbor in final_contours:  # final_contours存储了符合条件的轮廓的(x,y,w,h)值
                (x2, y2, w2, h2) = neighbor
                for (x3, y3) in neighbor_positions:  # 位置从左至右，从上到下
                    # neighbor_position位于每个轮廓线的中心，而不是左上角
                    # logic: (top left < center pos) and (bottom right > center pos)
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        contour_neighbors[index].append(neighbor)
        return contour_neighbors

    def image_preprocess(self, Image, threshold_min, threshold_max, imgname):
        grayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)  # RGB转灰度图
        blurredImage = cv2.blur(grayImage, (3, 3))  # 均值滤波
        # blurredImage = cv2.GaussianBlur(grayImage, (3, 3), 0)      #高斯模糊扩大边缘效果
        cannyImage = cv2.Canny(blurredImage, threshold_min, threshold_max, 3)  # 边缘检测，阈值[30,60]
        imshow_name = "cannyImage" + imgname
        cv2.imshow(imshow_name, cannyImage)
        cv2.waitKey(1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 返回指定形状和尺寸的结构元素,矩形：MORPH_RECT;内核矩阵：(9, 9))
        # kernel = np.ones((3,3), np.uint8)
        dilatedImage = cv2.dilate(cannyImage, kernel)  # 边缘膨胀
        # dilatedImage = cv2.dilate(cannyImage, kernel, iterations=2)
        imshow_name = "dilatedImage" + imgname
        cv2.imshow(imshow_name, dilatedImage)
        cv2.waitKey(1)

        # erodedImage = cv2.erode(dilatedImage,kernel)
        # cv2.imshow("erodedImage", erodedImage)
        # cv2.waitKey(1)

        return dilatedImage

    def find_roi(self, dilatedImage, Image):
        # CV_RETR_TREE:检测所有轮廓;CHAIN_APPROX_SIMPLE：压缩轮廓，把横竖撇捺都压缩得只剩下顶点
        (contours, hierarchy) = cv2.findContours(dilatedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 检测出物体的轮廓
        final_contours = []
        final_contours_Area = []
        final_box = []
        Image_box = Image.copy()
        """不再寻找单个魔方面的9个色块，改为直接寻找魔方面，直接提取目标魔方面的9个色块区域，即ROI，
           然后直接进行颜色识别，可达到一次性完成色块提取和颜色识别，但同样也有缺陷，依赖魔方面9个色
           块的位置特征，如果魔方面倾斜角度过大，可能需要进行仿射变换"""
        # 步骤1：过滤所有的轮廓，选择出符合给定条件的矩形/方形轮廓
        for contour in contours:  # 遍历所有轮廓
            perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter,
                                      True)  # 对图像轮廓点进行多边形拟合(输入的点集,精度：即是原始曲线与近似曲线之间的最大距离，若为true，则说明近似曲线是闭合的)
            # (x, y, w, h) = cv2.boundingRect(approx)  # 计算轮廓的包围框，并使用包围框计算高宽比(得到包覆此轮廓的最小正矩形)
            # cv2.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Image1", Image)
            # cv2.waitKey(1)
            # M = cv2.moments(contour)                                #计算轮廓特征矩
            # cX = int(M["m10"] / M["m00"])                           #cX，cY为轮廓质心
            # cY = int(M["m01"] / M["m00"])
            # Hu_M = cv2.HuMoments(contour)                         #不变矩，对平移、缩放、镜像和旋转都不敏感
            corners = len(approx)  # 多边形的角/顶点
            if corners == 4:  # 判断轮廓是否为矩形(矩形有四个角)
                area = cv2.contourArea(contour)  # 计算轮廓面积
                # (x, y, w, h) = cv2.boundingRect(approx)  # 计算轮廓的包围框，并使用包围框计算高宽比(得到包覆此轮廓的最小正矩形)
                rect = cv2.minAreaRect(approx)
                ((x, y), w, h) = (rect[0], rect[1][0], rect[1][1])  # (x, y)为中心坐标

                ratio_lowlim, ratio_uplim = 0.8, 1.2  # 魔方面长宽比上下限
                area_lowlim, area_uplim = 4000, 14000  # 魔方面面积上下限

                ratio = w / float(h)  # 计算边界矩形在轮廓周围的长宽比
                if ratio_lowlim <= ratio <= ratio_uplim and area / (
                        w * h) > 0.4 and area_lowlim < area < area_uplim:  # 判断轮廓的包围框是否接近正方形，根据实际情况更改正方形/矩形判断条件
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # Image_box = Image.copy()
                    # cv2.drawContours(Image_box, [box], 0, (0, 255, 0), 3)
                    # cv2.imshow("Image_approx", Image_box)
                    # cv2.waitKey(1)
                    # plt.imshow(Image[:, :, ::-1])
                    # plt.show()
                    final_contours.append((x, y, w, h))  # 增加元素
                    final_box.append(box)
                    final_contours_Area.append(area)

        if len(final_contours) < 1:
            return [], [], []
        window_cx, window_cy = 640 / 2, 480 / 2
        R = []  # 色块与摄像头视野中心坐标(640/2,480/2)距离
        for (x, y, w, h) in final_contours:
            temp = math.sqrt((x - window_cx) ** 2 + (y - window_cy) ** 2)
            R.append(temp)

        if min(R) > 25 * 1.5:
            return [], [], []

        index = R.index(min(R))
        (cx, cy, w, h) = final_contours[index]  # 寻找到魔方面
        """修改：以魔方面中心色块为基准，依靠魔方几何特征计算另外8个色块,中心色块选取规则为距离摄像头视野中心坐标(640/2,480/2)最近的色块"""
        center_box = final_box[index]  # 中心色块四个顶点坐标,顺时针：left、up、right、down

        box = []
        for (x, y) in center_box:
            coord_x = x - (cx - x) * 2 * 1.5
            coord_y = y - (cy - y) * 2 * 1.5
            box.append((coord_x, coord_y))

        # print(np.int0(box))  # 魔方面的顶点坐标
        corner_cube_cxy = []
        for (x, y) in box:  # 魔方面四个角方块的中心坐标，依赖几何特征
            # k = (y - cy)/(x - cx)
            # b = cy - k*cx
            cube_cx = x + (cx - x) / 3
            cube_cy = y + (cy - y) / 3
            corner_cube_cxy.append((cube_cx, cube_cy))
        # print(np.int0(corner_cube_cxy))
        side_corner_cube_cxy = []
        for i in range(len(corner_cube_cxy)):  # 魔方面每两个角方块之间的方块的中心坐标，依赖几何特征
            if i == len(corner_cube_cxy) - 1:
                (cx1, cy1) = corner_cube_cxy[i]
                (cx2, cy2) = corner_cube_cxy[0]
            else:
                (cx1, cy1) = corner_cube_cxy[i]
                (cx2, cy2) = corner_cube_cxy[i + 1]

            cube_cx = cx1 + (cx2 - cx1) / 2
            cube_cy = cy1 + (cy2 - cy1) / 2

            side_corner_cube_cxy.append((cube_cx, cube_cy))
        # print(np.int0(side_corner_cube_cxy))
        # cube_positions = corner_cube_cxy + side_corner_cube_cxy
        # cube_positions.append((cx, cy))
        # plt.imshow(Image[:, :, ::-1])
        # plt.show()
        # final_cube_positions = self.cube_sort(cube_positions)
        # 对色块排序，以中心色块顶点排列为准
        final_cube_positions = [corner_cube_cxy[0], side_corner_cube_cxy[0], corner_cube_cxy[1],
                                side_corner_cube_cxy[3], (cx, cy), side_corner_cube_cxy[1],
                                corner_cube_cxy[3], side_corner_cube_cxy[2], corner_cube_cxy[2]]

        # final_cube_positions = np.int0(final_cube_positions)
        # # rx, ry = w/3, h/3
        # # [ (x-rx,y-ry),    (x,y-ry),     (x+rx,y-ry)
        # #  (x-rx,y)    ,    (x,y)    ,     (x+rx,y)
        # #  (x-rx,y+ry),    (x,y+ry),     (x+rx,y+ry) ]
        # center_x = x + w / 2  # 计算魔方面的中心(x,y)坐标
        # center_y = y + h / 2
        # rx, ry = w / 3, h / 3
        # cube_positions = [
        #     # top left                          top middle                      top right
        #     [(center_x - rx), (center_y - ry)], [center_x, (center_y - ry)], [(center_x + rx), (center_y - ry)],
        #     # middle left                       center                          middle right
        #     [(center_x - rx), center_y],        [center_x, center_y],        [(center_x + rx), center_y],
        #     # bottom left                       bottom middle                   bottom right
        #     [(center_x - rx), (center_y + ry)], [center_x, (center_y + ry)], [(center_x + rx), (center_y + ry)],
        # ]
        roi = []
        image_roi = Image.copy()
        try:
            temp = 2
            for (cx, cy) in final_cube_positions:
                cx, cy = int(cx), int(cy)
                roi.append(Image[cy - 12 * temp:cy + 13 * temp, cx - 12 * temp:cx + 13 * temp])  # roi = img(25*25)
                cv2.rectangle(image_roi, (cx - 12 * temp, cy - 12 * temp), (cx + 13 * temp, cy + 13 * temp),
                              (0, 255, 0), 2)
                # cv2.imshow('image_roi', image_roi)
                # cv2.waitKey(1)
        except:
            return [], [], []
        return roi, image_roi, final_cube_positions

    def image_PerspectiveTransform(self, Image, coord, cube_len):
        """仿射变换-透视变换"""
        width, height = cube_len, cube_len
        # 设置特征图像和生成图像的坐标
        src = np.float32(coord)  # 左上，右上，左下，右下4个坐标
        dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 通过运算得出M矩阵
        M = cv2.getPerspectiveTransform(src, dst)
        # 提取特征图片
        OutImage = cv2.warpPerspective(Image, M, (int(width), int(height)))
        return OutImage

    def find_roi_twosides(self, dilatedImage, Image, sides_position):
        # CV_RETR_TREE:检测所有轮廓;CHAIN_APPROX_SIMPLE：压缩轮廓，把横竖撇捺都压缩得只剩下顶点
        (contours, hierarchy) = cv2.findContours(dilatedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 检测出物体的轮廓
        # cv2.drawContours(Image,contours,-1,(0,0,255),3)
        # cv2.imshow("ContoursImage", Image)
        # cv2.waitKey(1)

        # 仿射变换-透视变换
        # plt.imshow(Image[:, :, ::-1])
        # plt.show()
        x_axis_len, y_axis_len = 640, 480
        if sides_position == 'up':
            xstart, xend = 117, 512
            cube_len = xend - xstart
            coord1_right = [(166, 23), (465, 23), (xstart, y_axis_len / 2), (xend, y_axis_len / 2)]  # 左上，右上，左下，右下4个坐标
            coord2_left = [(xstart, y_axis_len / 2), (xend, y_axis_len / 2), (166, y_axis_len - 23),
                           (465, y_axis_len - 23)]
            # coord1_right = [(177, 29), (461, 23), (131, 210), (505, 205)]  # 左上，右上，左下，右下4个坐标
            # coord2_left = [(129, 234), (505, 225), (171, 442),(462, 444)]

        elif sides_position == 'down':
            xstart, xend = 120, 492
            cube_len = xend - xstart
            coord1_right = [(180, 35), (460, 35), (xstart, y_axis_len / 2), (xend, y_axis_len / 2)]  # 左上，右上，左下，右下4个坐标
            coord2_left = [(xstart, y_axis_len / 2), (xend, y_axis_len / 2), (180, 438), (452, 445)]

        # DilatedImage_right = self.image_PerspectiveTransform(dilatedImage.copy(), coord1_right, cube_len)
        # DilatedImage_left = self.image_PerspectiveTransform(dilatedImage.copy(), coord2_left, cube_len)
        Image_right = self.image_PerspectiveTransform(Image.copy(), coord1_right, cube_len)
        Image_left = self.image_PerspectiveTransform(Image.copy(), coord2_left, cube_len)
        # cv2.imshow('DilatedImage_right', Image_right)
        # cv2.waitKey(1)
        # cv2.imshow('DilatedImage_left', Image_left)
        # cv2.waitKey(1)

        cx, cy = cube_len / 2, cube_len / 2
        box = [(0, 0), (cube_len, 0), (cube_len, cube_len), (0, cube_len), ]  # 魔方面四个顶点坐标(顺时针)
        # print(np.int0(box))
        corner_cube_cxy = []
        for (x, y) in box:  # 魔方面四个角方块的中心坐标，依赖几何特征
            # k = (y - cy)/(x - cx)
            # b = cy - k*cx
            cube_cx = x + (cx - x) / 3
            cube_cy = y + (cy - y) / 3
            corner_cube_cxy.append((cube_cx, cube_cy))
        # print(np.int0(corner_cube_cxy))
        side_corner_cube_cxy = []
        for i in range(len(corner_cube_cxy)):  # 魔方面每两个角方块之间的方块的中心坐标，依赖几何特征
            if i == len(corner_cube_cxy) - 1:
                (cx1, cy1) = corner_cube_cxy[i]
                (cx2, cy2) = corner_cube_cxy[0]
            else:
                (cx1, cy1) = corner_cube_cxy[i]
                (cx2, cy2) = corner_cube_cxy[i + 1]

            cube_cx = cx1 + (cx2 - cx1) / 2
            cube_cy = cy1 + (cy2 - cy1) / 2

            side_corner_cube_cxy.append((cube_cx, cube_cy))
        # print(np.int0(side_corner_cube_cxy))
        cube_positions = corner_cube_cxy + side_corner_cube_cxy
        cube_positions.append((cx, cy))
        # plt.imshow(Image[:, :, ::-1])
        # plt.show()
        final_cube_positions = self.cube_sort(cube_positions)
        TwoImage = [Image_right, Image_left]
        TwoImage_roishow = [Image_right.copy(), Image_left.copy()]
        roi = []
        for index, img in enumerate(TwoImage):  # 获取图片ROI区域
            try:
                temp = 2
                for (cx, cy) in final_cube_positions:
                    cx, cy = int(cx), int(cy)
                    roi.append(img[cy - 12 * temp:cy + 13 * temp, cx - 12 * temp:cx + 13 * temp])  # roi = img(25*25)，原图
                    cv2.rectangle(TwoImage_roishow[index], (cx - 12 * temp, cy - 12 * temp),
                                  (cx + 13 * temp, cy + 13 * temp), (0, 255, 0), 2)
                    # cv2.imshow('image_roi', img)
                    # cv2.waitKey(1)
            except:
                return [], [], []
        # TwoImage = np.vstack((Image_right, Image_left))  # 矩阵列连接
        # inputs = np.hstack((Image_right, Image_left)) # 矩阵行连接
        # cv2.imshow('input img', TwoImage)
        # cv2.waitKey(1)
        return roi, TwoImage_roishow, final_cube_positions

    def convert_bgr_to_notation(self, bgr):
        """
        将BGR元组转换为魔方符号。
        :param bgr tuple: The BGR values to convert.
        :returns: str
        """
        notations = {
            'green': 'F',
            'white': 'U',
            'blue': 'B',
            'red': 'R',
            'orange': 'L',
            'yellow': 'D'
        }
        color_name = self.get_closest_color(bgr)['color_name']
        return notations[color_name]

    def get_result_notation(self, cube_state):
        """将所有的面及其BGR颜色转换为魔方符号"""
        notations = {
            'green': 'F',
            'white': 'U',
            'blue': 'B',
            'red': 'R',
            'orange': 'L',
            'yellow': 'D'
        }
        notation = dict(cube_state)
        for side, preview in notation.items():
            for sticker_index, color_name in enumerate(preview):
                # notation[side][sticker_index] = self.convert_bgr_to_notation(bgr)   # 每个面的9个色块bgr值转为魔方符号
                notation[side][sticker_index] = notations[color_name]

        # 将6个面的所有色块对应的魔方符号连接成一个字符串以满足输入要求
        # 顺序是URFDLB(white, red, green, yellow, orange, blue)(白，红，绿，黄，橙，蓝)
        combined = ''
        for side in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:  # URFDLB
            combined += ''.join(notation[side])  # . join()：将序列（也就是字符串、元组、列表、字典）中的元素以指定的字符连接生成一个新的字符串。
        return combined

    def nothing(self, x):
        pass

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(cap.get(cv2.CAP_PROP_EXPOSURE))
        # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频帧数
        # cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 关闭自动白平衡
        # print(cap.set(cv2.CAP_PROP_AUTO_WB, 0))
        # 设置曝光为手动模式
        # print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25))
        # 设置曝光的值为0
        # print(cap.set(cv2.CAP_PROP_EXPOSURE, 100))

        # cv2.namedWindow('Trackbar_canny')
        # cv2.createTrackbar('min', 'Trackbar_canny', 0, 1000, self.nothing)
        # cv2.createTrackbar('max', 'Trackbar_canny', 0, 1000, self.nothing)
        while cap.isOpened():
            if _CubeRobotUiDesigner.isExitThread:  # 结束线程标志
                cap.release()
                break

            _, Image = cap.read()
            title = "OriginImage" + str(self.camera_id)
            cv2.imshow(title, Image)
            cv2.waitKey(1)
            global_mysignals.showImageSignal.emit(tuple(Image), self.sides_position, self.detection_mode, self.camera_id)
            if _CubeRobotUiDesigner.isStartSolve:  # 启动识别
                # 重写线程，以下为图像处理
                """图像预处理"""
                # threshold_min = cv2.getTrackbarPos('min', 'Trackbar_canny')
                # threshold_max = cv2.getTrackbarPos('max', 'Trackbar_canny')
                originname = "origin_ID_" + str(self.camera_id)
                dilatedImage = self.image_preprocess(Image, 30, 60, originname)  # [30, 60],[20,0]
                """检测魔方色块轮廓"""

                if self.detection_mode == 0:  # 单面检测
                    roi, imgshow_roi, roi_cxy = self.find_roi(dilatedImage, Image)
                else:  # 双面检测
                    roi, imgshow_roi, roi_cxy = self.find_roi_twosides(dilatedImage, Image, self.sides_position)

                if len(roi) == 9 or len(roi) == 18:
                    image_n = roi.copy()
                    colors_name = CubeSvm().get_closest_color_from_svm_model(image_n)
                    result_colors_name = []  # 当前图片色块颜色名
                    [result_colors_name.extend(value_name) for value_name in list(colors_name)]

                    if len(roi) == 9:
                        lock.acquire()  # 上锁
                        if self.sides_position == 'front':
                            result_state['R'] = result_colors_name.copy()
                        elif self.sides_position == 'back':
                            result_state['L'] = result_colors_name.copy()
                            array_dir, array_src = [7, 4, 1, 2, 3, 6, 9, 8], [1, 2, 3, 6, 9, 8, 7, 4]
                            for index, src in enumerate(array_src):
                                result_state['L'][array_dir[index] - 1] = result_colors_name[src - 1]
                        lock.release()  # 解锁
                        for index, (x, y) in enumerate(roi_cxy):
                            x, y = int(x - 25), int(y - 25)
                            cv2.putText(imgshow_roi, result_colors_name[index], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (0, 0, 255), 2)

                        name = "Image_colors_one_ID_" + str(self.camera_id)
                    elif len(roi) == 18:  # 两个面18个方块，中心方块序列号为：4，13，从左到右，从上到下
                        lock.acquire()  # 上锁
                        if self.sides_position == 'up':
                            result_state['U'] = result_colors_name[0:9].copy()
                            result_state['U'].reverse()  # U(1)面倒序
                            result_state['B'] = result_colors_name[9:18].copy()
                        elif self.sides_position == 'down':  # F(0)、D(3)
                            result_state['F'] = result_colors_name[0:9].copy()
                            result_state['D'] = result_colors_name[9:18].copy()
                        lock.release()  # 解锁
                        for img_index, img in enumerate(imgshow_roi):
                            for i, (x, y) in enumerate(roi_cxy):
                                x, y = int(x - 25), int(y - 25)
                                cv2.putText(img, result_colors_name[i + img_index * 9], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.75, (0, 0, 255), 2)

                        name = "Image_colors_two_ID_" + str(self.camera_id)
                        imgshow_roi = np.vstack((imgshow_roi[0], imgshow_roi[1]))  # 矩阵列连接
                        # TwoImage = np.hstack((Image_right, Image_left)) # 矩阵行连接

                    cv2.imshow(name, imgshow_roi)
                    cv2.waitKey(1)
                    # print(result_state)
                    # return result_state

                    self.preview_colors_name = {}
                    self.average_sticker_colors_name = {}
                    self.average_sticker_colors = {}  # 魔方色块颜色RGB存储
                    self.preview_state = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                                          (255, 255, 255), (255, 255, 255), (255, 255, 255),
                                          (255, 255, 255), (255, 255, 255), (255, 255, 255)]  # 可更新



class CubeSvm:
    def __init__(self):
        # self.dirfile_path = 'C:/Users/35021/Desktop/Rubik Cube robot/SVM/cube_train_image'  # 存放6个颜色文件夹的文件路径(即总数据文件夹路径)
        # self.model_path = 'C:/Users/35021/Desktop/Rubik Cube robot/SVM/svm_cube.model'  # 模型保存路径
        self.dirfile_path = 'cube_train_image'  # 存放6个颜色文件夹的文件路径(即总数据文件夹路径)
        self.model_path = 'svm_cube.model'  # 模型保存路径

    def get_filelist(self, file_abspath):
        filelist = []  # 训练数据文件列表
        for filename in os.listdir(file_abspath):  # os.listdir()用于返回一个由文件名和目录名组成的列表,输入为绝对路径
            filepath = os.path.join(file_abspath + '/', filename)
            if filepath.endswith('.jpg'):  # 判断字符串是否以指定字符或子字符串结尾，常用于判断文件类型
                filelist.append(filepath)
        return filelist

    def get_dataMatandLabel(self, traindata_pathlist):
        dataLabel = []
        img_vectorlen = 1 * 1875 * 4
        dataNum = len(traindata_pathlist)
        dataMat = np.zeros((dataNum, img_vectorlen))
        for i, imgfile_path in enumerate(traindata_pathlist):
            imgName = os.path.split(imgfile_path)[1]  # 得到 颜色_编号.jpg     ['abspath','filename.jpg']
            classTag = imgName.split(".")[0].split("_")[0]  # 得到 类标签(颜色) 如  B_5.jpg => B
            dataLabel.append(classTag)
            image = cv2.imread(imgfile_path, 1)  # 1为默认参数，读入一副彩色图片
            dataMat[i, :] = self.img2vector(image)  # 行向量
        return dataMat, dataLabel

    def get_traindata(self, dirfile_path):
        filename_list = ['white', 'red', 'green', 'yellow', 'orange', 'blue']  # 共6个文件夹
        for i, filename in enumerate(filename_list):
            traindata_abspath = os.path.join(dirfile_path + '/', filename)
            traindata_pathlist = self.get_filelist(traindata_abspath)  # 获取该文件夹下所有图片路径列表

            if i == 0:
                dataMat, dataLabel = self.get_dataMatandLabel(traindata_pathlist)  # 该类下的数据集[imgvector * imgNum, 颜色]
            else:
                dataMat_, dataLabel_ = self.get_dataMatandLabel(traindata_pathlist)  # 该类下的数据集[imgvector * imgNum, 颜色]
                dataMat = np.concatenate((dataMat, dataMat_), axis=0)  # 设置axis=0则代表着按照第一维度进行拼接(即列不变，拼接行)
                dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
        return dataMat, dataLabel  # 6个类的总数据集和标签

    def img2vector(self, image):  # 将图像转换为向量
        img_normalize = np.array(image) / 255  # 图像归一化
        img_vector = np.reshape(img_normalize, (1, -1))  # reshape(1,-1)代表将二维数组重整为一个一行的二维数组(1 * 1875 向量)
        return img_vector

    def creat_svm(self, dataMat, dataLabel, model_path):
        classifier = svm.SVC(C=1.0, kernel='rbf')  # 创建svm分类器
        rf = classifier.fit(dataMat, dataLabel)  # 开始训练模型
        joblib.dump(rf, model_path)  # 存储训练好的模型
        return classifier

    def creat_traindata(self, image_n, centercube_colorname, group_index):
        # filename_list = ['white', 'red', 'green', 'yellow', 'orange', 'blue']
        for index, img in enumerate(image_n):
            imgName = '_'.join([centercube_colorname, str(group_index) + str(index)])  # 文件名，eg: white_00
            imgName += '.jpg'  # 文件格式，eg: white_00.jpg
            traindata_path = self.dirfile_path + '/' + centercube_colorname + '/'
            cv2.imwrite(os.path.join(traindata_path, imgName), img)

    def creat_svm_color_model(self):
        # start = time.clock()
        dataMat, dataLabel = self.get_traindata(self.dirfile_path)
        self.creat_svm(dataMat, dataLabel, self.model_path)
        # end = time.clock()
        # spent_time = format((end - start))
        # print("Training spent {:.4f}s.".format((end - start)))

    def get_closest_color_from_svm_model(self, dir_img_n):
        preResult = []
        for index, img in enumerate(dir_img_n):
            img2arr = self.img2vector(img)
            clf = joblib.load(self.model_path)  # 加载模型,model_path = '模型保存路径'
            preResult.append(clf.predict(img2arr))  # 得到颜色标签['white',...]
        return preResult


class mythread_solve_cube(threading.Thread):
    def __init__(self):
        super(mythread_solve_cube, self).__init__()
        self.cube_solve = []
        self.res = []
        self.select_mechanical_arm = []
        self.min_mechanical_steps = {}
        self.min_mechanical_steps_len = 0
        self.max_mechanical_steps_len = 0
        self.send_mechanical_steps = ''
        # global result_state  # 全局变量
        # global cube_solve, res, select_mechanical_arm

    def get_result_notation(self, cube_state):
        """将所有的面及其BGR颜色转换为魔方符号"""
        notations = {
            'F': 'color',
            'U': 'color',
            'B': 'color',
            'R': 'color',
            'L': 'color',
            'D': 'color'
        }
        for side, preview in cube_state.items():
            notations[side] = preview[4]  # 每个面的中心方块颜色名

        # notation = dict(cube_state)       # 浅拷贝，指的是重新分配一块内存，创建一个新的对象，但里面的元素是原对象中各个子对象的引用
        notation = deepcopy(cube_state)  # 深拷贝，是指重新分配一块内存，创建一个新的对象，并且将原对象中的元素，以递归的方式，通过创建新的子对象拷贝到新对象中。因此，新对象和原对象没有任何关联。
        for side, preview in notation.items():
            for sticker_index, color_name in enumerate(preview):
                # notation[side][sticker_index] = self.convert_bgr_to_notation(bgr)   # 每个面的9个色块bgr值转为魔方符号
                notation[side][sticker_index] = list(notations.keys())[
                    list(notations.values()).index(color_name)]  # 依靠value索引key值

        # 将6个面的所有色块对应的魔方符号连接成一个字符串以满足输入要求
        # 顺序是URFDLB(white, red, green, yellow, orange, blue)(白，红，绿，黄，橙，蓝)
        combined = ''
        for side in ['U', 'R', 'F', 'D', 'L', 'B']:  # URFDLB
            combined += ''.join(notation[side])  # . join()：将序列（也就是字符串、元组、列表、字典）中的元素以指定的字符连接生成一个新的字符串。
        return combined

    # def get_mechanical_steps(self, cube_solve_steps):
    #     # 定义魔方坐标系 (0)[left_down]、(1)[left_up]、(2)[front]、(3)[right_down]、(4)[back]、(5)[right_up]
    #     # (0)：左夹爪正对面；(3)：右夹爪正对面
    #     # (5)：F(0)对面；(1)：D(3)对面
    #     # (2)：前面；(4)：后面
    #
    #     # 机械步骤转换，例如：cube_solve_steps = ["B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'"]
    #     length = len(cube_solve_steps.split(' '))
    #     solve_steps = cube_solve_steps.split(' ')
    #     control_cube_sides = {  # 需要转动某一个魔方面时先执行的机械步骤，夹爪夹住后可执行具体转动步骤(90°,90°',180°)
    #         'U': ['RO', 'L2', 'RC'],
    #         'R': ['RO', 'L', 'RC'],
    #         'F': ['L'],
    #         'D': ['R'],
    #         'L': ['LO', 'R', 'LC'],
    #         'B': ['LO', 'R2', 'LC']
    #     }
    #     initial_csys = ['F', 'U', 'R', 'D', 'L',
    #                     'B']  # 构建模方坐标系,初始坐标系：F(0)[left_down]、U(1)[left_up]、R(2)[front]、D(3)[right_down]、L(4)[back]、B(5)[right_up]
    #     csys_transform = {  # 魔方坐标系变换，coordinate system : csys
    #         # trun:
    #         'U': ['F', 'D', 'L', 'U', 'R', 'B'],
    #         'B': ['B', 'U', 'L', 'D', 'R', 'F'],
    #         'left_trun': {
    #             'R': ['F', 'L', 'U', 'R', 'D', 'B'],
    #             'L': ['F', 'R', 'D', 'L', 'U', 'B']
    #         },
    #         'right_trun': {
    #             'R': ['R', 'U', 'B', 'D', 'F', 'L'],
    #             'L': ['L', 'U', 'F', 'D', 'B', 'R']
    #         }
    #     }
    #     mechanical_steps = []
    #     last_cube_csys, updata_cube_csys = initial_csys.copy(), []
    #     for index, steps in enumerate(solve_steps):
    #         # mechanical_steps[steps] = control_cube_sides[steps[0]].copy()
    #         # if len(steps) == 2:
    #         #     mechanical_steps[steps].append(control_cube_sides[steps[0]][-1][0] + steps[1])
    #         # 不考虑(1)、(5)坐标面分别选择左机械臂和右机械臂转动所带来的多余机械步骤
    #         # 对(2)、(4)坐标面选择左机械臂转面还是右机械臂转面进行全局规划
    #         for encode_index in [initial_csys.index(value) for value in csys_transform[steps[0]]]:
    #             updata_cube_csys.append(last_cube_csys[encode_index])
    #         # updata_cube_csys.append(last_cube_csys[#的序列编号]) #的序列编号 = turn[U]的序列编号
    #
    #         # cube_side = initial_csys[last_cube_csys.index(steps)]
    #         mechanical_steps[index] = control_cube_sides[initial_csys[last_cube_csys.index(
    #             steps)]].copy()  # 映射面：initial_csys[last_cube_csys.index(steps)] ， 上个坐标系的某个面映射到初始坐标系的某个面
    #         if len(steps) == 2:
    #             mechanical_steps[index].append(initial_csys[last_cube_csys.index(steps)][-1][0] + steps[1])
    #
    #         last_cube_csys = updata_cube_csys.copy()
    #         updata_cube_csys = []
    def updata_cube_csys(self, turn_cubeside, last_cube_csys, mechanical_arm):  # 更新魔方坐标系
        initial_csys = ['F', 'U', 'R', 'D', 'L', 'B']
        csys_transform = {  # 魔方坐标系变换，coordinate system : csys
            # trun:
            'U': ['F', 'D', 'L', 'U', 'R', 'B'],
            'B': ['B', 'U', 'L', 'D', 'R', 'F'],
            'left_arm': {
                'R': ['F', 'L', 'U', 'R', 'D', 'B'],
                'L': ['F', 'R', 'D', 'L', 'U', 'B']
            },
            'right_arm': {
                'R': ['R', 'U', 'B', 'D', 'F', 'L'],
                'L': ['L', 'U', 'F', 'D', 'B', 'R']
            },
            'F': ['F', 'U', 'R', 'D', 'L', 'B'],
            'D': ['F', 'U', 'R', 'D', 'L', 'B']
        }
        new_cube_csys = []
        initial_csys_surface_of_mapping = initial_csys[last_cube_csys.index(
            turn_cubeside[0])]  # 映射面：initial_csys[last_cube_csys.index(steps)]，上一个坐标系的某个面映射到初始坐标系的某个面,即机械实际转动面

        if initial_csys_surface_of_mapping == 'R' or initial_csys_surface_of_mapping == 'L':  # 只(2)、(4)坐标面选择左机械臂转面还是右机械臂转面产生的不同坐标面变换
            _csys_transform = csys_transform[mechanical_arm][initial_csys_surface_of_mapping]
        else:
            _csys_transform = csys_transform[initial_csys_surface_of_mapping]

        encode = [initial_csys.index(value) for value in _csys_transform]
        for encode_index in [initial_csys.index(value) for value in _csys_transform]:  # 坐标系转换
            new_cube_csys.append(last_cube_csys[encode_index])

        if len(turn_cubeside) == 2:
            initial_csys_surface_of_mapping += turn_cubeside[1]
        return initial_csys_surface_of_mapping, new_cube_csys

        # 深度优先搜索(DFS)思路：
        # 1、创建结果存储变量，初始化当前结果
        # 2、设计递归函数：
        #    函数执行过程：
        #    -若到达结尾，则返回return
        #    -若未到达结尾，则更新当前结果
        #    -若到达末尾叶子节点，进行最优结果更新
        #    -分别对当前节点的左/右叶子节点调用递归函数
        # 3、开始调用递归函数
        '''
        def dfsTemplate():
            # 创建结果存储变量
            res = []
            # 初始化当前结果
            start = []
            def dfs(node, currentResult):
                # 若到达结尾，则返回return
                if(node == null): # 终止条件判断
                    return

        '''

    def dfs(self, node, node_addsave, mechanical_arm, dir_save, depth, last_cube_csys):
        node_addsave[depth] = node  # 记录当前枝节点
        # -终止条件，若到达结尾，则返回return
        if mechanical_arm == 'left_arm' and node[
            0] == 'B':  # 如果为：'B'，若为left:return，right臂转动魔方,即左手不能转动B坐标面(整体)，终止左手转动B坐标面的分支
            return
        elif mechanical_arm == 'right_arm' and node[0] == 'U':  # 如果为：'U'，若为right:return，left臂转动魔方
            return
        elif mechanical_arm == 'left_arm' and node[0] == 'F':
            return
        elif mechanical_arm == 'right_arm' and node[0] == 'D':
            return
        elif depth >= (len(self.cube_solve) - 1):  # 终止条件为：分支节点到达结尾
            return

        # -若未到达结尾，则更新当前结果
        dir_save[depth] = mechanical_arm
        real_turn_cubeside, updata_cube_csys = self.updata_cube_csys(self.cube_solve[depth], last_cube_csys,
                                                                     mechanical_arm)  # 机械实际转动面,更新魔方坐标系
        # node = real_turn_cubeside  # 枝节点：实际机械转动面
        # node_addsave[depth] = node  # 记录枝节点
        depth += 1
        real_turn_cubeside, updata_cube_csys2 = self.updata_cube_csys(self.cube_solve[depth], updata_cube_csys,
                                                                      mechanical_arm)  # 机械实际转动面,更新魔方坐标系
        node = real_turn_cubeside  # 枝节点：实际机械转动面,计算下一个节点

        # -若到达末尾叶子节点，进行最优结果更新
        # if (node_left == None and node_right == None):
        if depth == (len(self.cube_solve) - 1):
            # updata res
            node_addsave[depth] = node
            # dir_save[depth] = mechanical_arm
            self.res.append(node_addsave.copy())
            # self.select_mechanical_arm.append(dir_save.copy())
            # （针对机械臂选择，对末尾枝节点进行异臂）改：
            if node[0] == 'U' or node[0] == 'D':
                dir_save[depth] = 'left_arm'
                self.select_mechanical_arm.append(dir_save.copy())
            elif node[0] == 'B' or node[0] == 'F':
                dir_save[depth] = 'right_arm'
                self.select_mechanical_arm.append(dir_save.copy())
            else:  # 否则异臂
                if dir_save[depth - 1] == 'left_arm':
                    dir_save[depth] = 'right_arm'
                    self.select_mechanical_arm.append(dir_save.copy())
                else:
                    dir_save[depth] = 'left_arm'
                    self.select_mechanical_arm.append(dir_save.copy())
            # # 针对机械臂选择，对末尾枝节点进行异臂
            # if dir_save[depth-1] == 'left_arm':
            #     dir_save[depth] = 'right_arm'
            #     self.select_mechanical_arm.append(dir_save.copy())
            # else:
            #     dir_save[depth] = 'left_arm'
            #     self.select_mechanical_arm.append(dir_save.copy())

        # 左右子树递归
        self.dfs(node, node_addsave, 'left_arm', dir_save, depth, updata_cube_csys)
        self.dfs(node, node_addsave, 'right_arm', dir_save, depth, updata_cube_csys)

    def dfsTemplate(self):
        self.res = []  # 清空列表
        self.select_mechanical_arm = []
        initial_csys = ['F', 'U', 'R', 'D', 'L', 'B']
        start_dapth = 0
        root = self.cube_solve[start_dapth]
        node_addsave = {}
        mechanical_arm_save = {}
        self.dfs(root, node_addsave, 'left_arm', mechanical_arm_save, start_dapth, initial_csys)
        self.dfs(root, node_addsave, 'right_arm', mechanical_arm_save, start_dapth, initial_csys)
        # print('二叉树每条分支：', self.res)
        # print(self.select_mechanical_arm)

    def get_mechanical_steps(self):
        # 定义魔方坐标系 (0)[left_down]、(1)[left_up]、(2)[front]、(3)[right_down]、(4)[back]、(5)[right_up]
        # (0)：左夹爪正对面；(3)：右夹爪正对面
        # (5)：F(0)对面；(1)：D(3)对面
        # (2)：前面；(4)：后面

        # 机械步骤转换，例如：cube_solve_steps = ["B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'"]
        # length = len(cube_solve_steps.split(' '))
        # solve_steps = cube_solve_steps.split(' ')
        # 若魔方面转动180°，夹爪无需执行归位打开，若转动90°，夹爪无需执行关闭操作
        control_mechanical_steps_90deg = {  # left_arm、right_arm为转动整个魔方的机械臂，而非转动具体某个面的机械臂
            'left_arm': {
                'D': ['R'],  # , 'RO', "R'"
                'U': ['RO', 'L2', 'RC', 'R'],  # , 'R0', "R'"
                'R': ['RO', 'L', 'RC', 'LO', "L'", 'LC', 'R'],  # , 'RO', "R'"
                'L': ['RO', "L'", 'RC', 'LO', "L", 'LC', 'R']  # , 'RO', "R'"
            },
            'right_arm': {
                'F': ['L'],  # , 'LO', "L'"
                'B': ['LO', 'R2', 'LC', 'L'],  # , 'L0', "L'"
                'R': ['LO', "R'", 'LC', 'RO', 'R', 'RC', 'L'],  # , 'LO', "L'"
                'L': ['LO', 'R', 'LC', 'RO', "R'", 'RC', 'L']  # , 'LO', "L'"
            }
        }
        control_mechanical_steps_180deg = {  # 需要转动某一个魔方面时先执行的机械步骤，夹爪夹住后可执行具体转动步骤(90°,90°',180°)
            'left_arm': {
                'D': ['R2'],
                'U': ['RO', 'L2', 'RC', 'R2'],
                'R': ['RO', 'L', 'RC', 'LO', "L'", 'LC', 'R2'],
                'L': ['RO', "L'", 'RC', 'LO', "L", 'LC', 'R2']
            },
            'right_arm': {
                'F': ['L2'],
                'B': ['LO', 'R2', 'LC', 'L2'],
                'R': ['LO', "R'", 'LC', 'RO', 'R', 'RC', 'L2'],
                'L': ['LO', 'R', 'LC', 'RO', "R'", 'RC', 'L2']
            }
        }
        binary_tree_mechanical_steps = {}

        for index, steps_dict in enumerate(self.res):
            cube_solve_steps = ' '.join(steps for steps in steps_dict.values())
            # if cube_solve_steps == "R2 U F' L2 U' R2 D2 R' D2 R' L B2 D2 F2 B2 L' R2 L2 L' L2":
            #     print('find')
            solve_steps_length = len(cube_solve_steps.split(' '))
            for key_value, steps in enumerate(cube_solve_steps.split(' ')):
                mechanical_steps = []
                # 末端节点处理
                if key_value == solve_steps_length - 1:
                    if len(cube_solve_steps.split(' ')) == 1:  # 若只有一个执行面则需进行复位
                        present_mechanical_arm = 'left_arm'
                        # present_mechanical_arm = 'right_arm'
                        if steps[-1] == "2":  # 顺时针180°
                            mechanical_steps = control_mechanical_steps_180deg[present_mechanical_arm][steps[0]].copy()
                        else:
                            mechanical_steps = control_mechanical_steps_90deg[present_mechanical_arm][steps[0]].copy()
                    else:  # 执行面大于1
                        last_steps = list(cube_solve_steps.split(' '))[key_value - 1]  # 上个执行面
                        present_mechanical_arm = self.select_mechanical_arm[index][key_value]  # 当前机械臂
                        if (steps[0] == 'R' or steps[0] == 'L') and last_steps[
                            -1] != "2":  # 若上一个执行面转90°，则当前执行面为R/L时，需异臂以延续上个面的执行步骤
                            if steps[-1] == "2":  # 顺时针180°
                                mechanical_steps = control_mechanical_steps_180deg[present_mechanical_arm][steps[0]][
                                                   0:3].copy() + [
                                                       control_mechanical_steps_180deg[present_mechanical_arm][
                                                           steps[0]][-1]].copy()
                            else:
                                mechanical_steps = control_mechanical_steps_90deg[present_mechanical_arm][steps[0]][
                                                   0:3].copy() + [
                                                       control_mechanical_steps_90deg[present_mechanical_arm][steps[0]][
                                                           -1]].copy()
                        else:
                            if steps[-1] == "2":  # 顺时针180°
                                mechanical_steps = control_mechanical_steps_180deg[present_mechanical_arm][
                                    steps[0]].copy()
                            else:
                                mechanical_steps = control_mechanical_steps_90deg[present_mechanical_arm][
                                    steps[0]].copy()
                    if steps[-1] != "2":  # 非180°
                        if steps[-1] == "'":  # 逆时针90°
                            mechanical_steps[-1] += "'"
                            add_reset_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0],
                                               mechanical_steps[-1][0] + "C"]  # 复位步骤
                        else:  # 顺时针90°
                            add_reset_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0] + "'",
                                               mechanical_steps[-1][0] + "C"]  # 复位步骤
                        mechanical_steps += add_reset_steps
                # 其余节点处理
                else:
                    present_steps = steps  # 当前执行面
                    next_steps = list(cube_solve_steps.split(' '))[key_value + 1]  # 下个执行面
                    present_mechanical_arm = self.select_mechanical_arm[index][key_value]  # 当前机械臂
                    next_mechanical_arm = self.select_mechanical_arm[index][key_value + 1]  # 下个机械臂
                    if key_value == 0:
                        last_steps = None
                        last_mechanical_arm = present_mechanical_arm
                    else:
                        last_steps = list(cube_solve_steps.split(' '))[key_value - 1]  # 上个执行面
                        last_mechanical_arm = self.select_mechanical_arm[index][key_value - 1]  # 上一个机械臂

                    # 若当前节点为R/L且为异臂节点，需要修改原有的机械步骤以减少多余步骤
                    if last_mechanical_arm != present_mechanical_arm and (steps[0] == 'R' or steps[0] == 'L') and \
                            last_steps[-1] != "2":
                        if steps[-1] == "2":  # 顺时针180°
                            mechanical_steps = control_mechanical_steps_180deg[present_mechanical_arm][steps[0]][
                                               0:3].copy() + [
                                                   control_mechanical_steps_180deg[present_mechanical_arm][steps[0]][
                                                       -1]].copy()
                        else:
                            mechanical_steps = control_mechanical_steps_90deg[present_mechanical_arm][steps[0]][
                                               0:3].copy() + [
                                                   control_mechanical_steps_90deg[present_mechanical_arm][steps[0]][
                                                       -1]].copy()
                    # 否则保持原有机械步骤
                    else:
                        if steps[-1] == "2":  # 顺时针180°
                            mechanical_steps = control_mechanical_steps_180deg[present_mechanical_arm][steps[0]].copy()
                        else:  # 顺/逆时针90°
                            mechanical_steps = control_mechanical_steps_90deg[present_mechanical_arm][
                                steps[0]].copy()  # 正常步骤
                    # 对原有步骤处理完后，需根据条件判断加上合适的后续机械步骤
                    # 若当前节点转动180°，则保持上面的步骤不变；
                    # 若当前节点转动90°，需根据条件判断加上合适的后续机械步骤，判断条件：
                    # 在当前节点需转动90°情况下，①、如果下一个节点需执行换臂操作且执行面为R/L时，夹爪无需复位，否则夹爪需要加上复位步骤；
                    #                       ②、如果下一个节点不执行换臂操作，则继续控制该机械臂，夹爪需要复位，原有步骤需加上复位步骤，但无需闭合
                    if steps[-1] != "2":
                        if present_mechanical_arm != next_mechanical_arm:  # 如果下一个节点需执行换臂操作，转动90°后，若下一个执行面为R/L时，无需复位，否则需要复位
                            if steps[-1] == "'":  # 逆时针90°
                                mechanical_steps[-1] += "'"  # 最后一个字符
                            if next_steps[0] != 'R' and next_steps[0] != 'L':  # 除R/L面外的面
                                if steps[-1] == "'":  # 逆时针90°
                                    add_reset_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0],
                                                       mechanical_steps[-1][0] + "C"]
                                else:  # 顺时针90°
                                    add_reset_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0] + "'",
                                                       mechanical_steps[-1][0] + "C"]
                                mechanical_steps += add_reset_steps
                        else:  # 如果不执行换臂操作，即继续控制该机械臂，夹爪需复位，但无需闭合
                            if steps[-1] == "'":  # 逆时针90°
                                mechanical_steps[-1] += "'"  # 最后一个字符
                                add_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0]]  # 复位步骤
                            else:  # 顺时针90°
                                add_steps = [mechanical_steps[-1][0] + 'O', mechanical_steps[-1][0] + "'"]  # 复位步骤
                            mechanical_steps += add_steps
                if key_value == 0:
                    binary_tree_mechanical_steps[cube_solve_steps] = [mechanical_steps]
                else:
                    binary_tree_mechanical_steps[cube_solve_steps].append(mechanical_steps)
        # print(binary_tree_mechanical_steps)
        for key, list_steps in binary_tree_mechanical_steps.items():
            key_len = len(key.split(' '))
            for index, mechanical_steps in enumerate(list_steps):
                if (index < key_len - 1) and len(mechanical_steps) > 2:
                    next_mechanical_steps = list_steps[index + 1]
                    if mechanical_steps[-2] == next_mechanical_steps[0]:
                        if mechanical_steps[-2][0] == mechanical_steps[-1][0]:
                            next_mechanical_steps.pop(0)

        all_mechanical_steps_len = {}
        for key, list_steps in binary_tree_mechanical_steps.items():  # 开始拼接
            list_extend = []
            [list_extend.extend(mechanical_steps) for mechanical_steps in list_steps]
            binary_tree_mechanical_steps[key] = list_extend.copy()
            all_mechanical_steps_len[key] = len(list_extend)  # 计算长度

        self.max_mechanical_steps_len = max(all_mechanical_steps_len.values())
        self.min_mechanical_steps_len = min(all_mechanical_steps_len.values())  # 获得机械步骤最少的值
        branch_index = []
        index = 0
        self.min_mechanical_steps = {}
        for key, steps_len in all_mechanical_steps_len.items():
            if steps_len == self.min_mechanical_steps_len:
                self.min_mechanical_steps[key] = binary_tree_mechanical_steps[key].copy()
                branch_index.append(index)
            index += 1
        # branch_index = []
        # binary_tree_mechanical_steps = {}   # 清空字典
        # count = {}
        # for key, mechanical_steps in self.min_mechanical_steps.items():
        #     index = 0
        #     for step in mechanical_steps:
        #         if step[-1] == "2":
        #             index += 1
        #     count[key] = index
        # print(self.min_mechanical_steps_len)

    def run(self):
        global result_state  # 全局变量
        global global_serial
        global flag_test
        while _CubeRobotUiDesigner.isStartSolve and not _CubeRobotUiDesigner.isExitThread:
            # time.sleep(0.1)
            try:
                state = self.get_result_notation(result_state)
                # state = "BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF"
                print(state)
                self.cube_solve = kociemba.solve(state)  # 魔方解算
                print(self.cube_solve)
                self.cube_solve = self.cube_solve.split(' ')
                self.dfsTemplate()
                # print(self.select_mechanical_arm)
                self.get_mechanical_steps()

                mechanical_steps = list(self.min_mechanical_steps.values())[0]  # 第一个value值
                # 对发送数据进行修改
                for index, step in enumerate(mechanical_steps):
                    if step[-1] == "'":  # 逆时针90°
                        mechanical_steps[index] = step[0] + "-"
                    elif len(step) == 1:  # 顺时针90°
                        mechanical_steps[index] += "1"

                self.send_mechanical_steps = ''.join(mechanical_steps)
                print(len(self.send_mechanical_steps))
                self.send_mechanical_steps = "#" + self.send_mechanical_steps + "%E%S"  # 发送步骤立马运行
                if flag_test == 0:
                    flag_test = 1
                    global_serial.serial_write_ascii(self.send_mechanical_steps)
                global_mysignals.printSerialDataSignal.emit("solve:" + '--'.join(self.cube_solve) + "----------------")  # 触发
                print(self.send_mechanical_steps)
                time.sleep(0.1)
                # self.send_mechanical_steps = ''
                # _serial.send_cube_solve(self.cube_solve)
            except Exception:
                print('solve error')
                global_mysignals.printSerialDataSignal.emit("solve error")
                time.sleep(0.1)     # 添加延时


flag_test = 0
# 串口通信
class serial_port_communication:
    def __init__(self, COM, baudrate):
        super(serial_port_communication, self).__init__()
        self.COM = COM
        self.Baudrate = baudrate
        '''
        port：读或者写端口
        baudrate：波特率
        bytesize：字节大小
        parity：校验位
        stopbits：停止位
        timeout：读超时设置
        writeTimeout：写超时
        xonxoff：软件流控
        rtscts：硬件流控
        dsrdtr：硬件流控
        interCharTimeout:字符间隔超时
        '''

    def list_serialports(self, OS='windows'):
        # 串口列表
        list_ports = list(serial.tools.list_ports.comports())  # 串口列表
        if len(list_ports) < 0:
            print('没有检测到串口')
        if OS == 'windows':
            list_ports_name = [str(comx.device) for comx in list_ports]
        else:
            list_ports_name = [str(comx.name) for comx in list_ports]
        return list_ports_name

    def open_serial(self):
        # 打开串口，默认8字节、无校验位、1位停止位、读超时设置为1s
        self.COM = serial.Serial(self.COM, self.Baudrate, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE,
                                 timeout=1)
        # self.COM.open()
        if self.COM.isOpen():
            print("串口打开成功！")
        else:
            print("串口打开失败！")
        return self.COM

    def serial_write_ascii(self, write_data):
        # 串口发送数据(ascii)
        # write_data = bytes().fromhex(value)  # 读取数据如果需要十六进制 则进行转换，如果是ascii 则可以直接打印，fromhex可以用来对输入进行直接bytes化
        # write_data = bytearray.fromhex(value.encode('hex'))
        try:
            self.COM.write(write_data.encode('utf-8'))
            # print(write_data.encode('utf-8'))
            # time.sleep(0.1)
            return True
        except BaseException as exception:
            print(exception)
            return False

    def serial_read_ascii(self):
        # 串口接收数据(ascii)
        try:
            data = self.COM.read(self.COM.in_waiting)
            return data
        except BaseException as exception:
            print(exception)
            return None

    def close_serial(self):
        self.COM.close()
        if self.COM.isOpen():
            print("串口关闭失败！")
        else:
            print("串口关闭成功！")

    def send_cube_solve(self, _solve):
        _solve = "B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'"
        self.serial_write_ascii(_solve)


# Qt建议只在主线程中操作界面，因为在另外一个线程直接操作界面，可能会导致意想不到的问题，比如：输出显示不全，甚至程序崩溃。
# 所以这时，推荐的方法是使用信号。
'''
    ①、自定义一个Qt的QObject类，里面封装一些自定义的Signal信号：
    一种信号定义为该类的一个静态属性，值为Signal 实例对象即可。可以定义多个Signal静态属性，对应这种类型的对象可以发出的多种信号。
    注意：Signal实例对象的初始化参数指定的类型，就是 发出信号对象时，传递的参数数据类型。因为Qt底层是C++开发的，必须指定类型。

    ②、定义主线程执行的函数处理Signal信号（通过connect方法）。

    ③、在新线程需要操作界面的时候，就通过自定义对象发出信号;
    通过该信号对象的 emit方法发出信号， emit方法的参数 传递必要的数据。参数类型 遵循 定义Signal时，指定的类型。

    ④、主线程信号处理函数，被触发执行，获取Signal里面的参数，执行必要的更新界面操作。
'''


# 自定义信号源对象类型，一定要继承自 QObject
class MySignals(QObject):
    # 定义一种信号，两个参数分别是： 操作控件和控件内容
    # 调用emit方法发信号时，传入参数必须是这里指定的参数类型
    showImageSignal = pyqtSignal(tuple, str, int, int) # str, int, list, dict, tuple
    printSerialDataSignal = pyqtSignal(str)
    displayRunTimeSignal = pyqtSignal(list)
    # 通过Signal 的 emit 触发执行 主线程里面的处理函数
    # emit参数和定义Signal的数量、类型必须一致
    # global_mysignals.showImageSignal.emit(imageToList, 'front', 'one', 1)  # 触发


class myLabel(QLabel):  # 继承QLabel
    def __init__(self):
        self.flag = False
        self.pen = QPen(Qt.red, 4,
                        Qt.SolidLine)  # QPen(Qt.black,2,Qt.SolidLine),钢笔,颜色设置为黑色，宽度设置为2像素，Qt.SolidLine是预定义的线条样式之一
        self.x1, self.y1 = None, None  # x1和y1是矩形左上角坐标
        self.x2, self.y2 = None, None  # x2和y2是矩形右下角坐标

    # 鼠标点击事件
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:  # 左键按下
            self.flag = True
            self.x1, self.y1 = event.x(), event.y()
            # self.setText("单击鼠标左键的事件: 自己定义")
            print("单击鼠标左键")  # 响应测试语句
        else:
            return
        # elif event.buttons() == Qt.RightButton:  # 右键按下
        #     self.setText("单击鼠标右键的事件: 自己定义")
        #     print("单击鼠标右键")  # 响应测试语句

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False
        print('mouse release')

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag == True:
            self.x2, self.y2 = event.x(), event.y()
            self.update()  # 鼠标移动的时候更新UI

    # 鼠标滚动事件，控制缩放
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # angleDelta() - 返回QPoint对象，为滚轮转过的数值，单位为1 / 8度。
        angleX = angle.x()
        angleY = angle.y()
        if angleY > 0:
            self.setText("滚轮向上滚动的事件: 自己定义")
            print("鼠标滚轮上滚")  # 响应测试语句
        else:  # 滚轮下滚
            self.setText("滚轮向下滚动的事件: 自己定义")
            print("鼠标滚轮下滚")  # 响应测试语句

    # 绘制操作在QWidget.paintEvent()中完成，绘制方法必须放在QtGui.QPainter对象的begin（）和end（）之间
    def paintEvent(self, event):  # 重写paintEvent方法
        super().paintEvent(event)  # 调用父类的paintEvent()，这个是为了显示你设置的效果。否则会是一片空白。
        width, hight = abs(self.x2 - self.x1), abs(self.y2 - self.y1)
        rectangle = QRect(QPoint(self.x1, self.y1), QSize(width,
                                                          hight))  # QRect类使用整数精度在平面中定义一个矩形。或QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        QPainter().setPen(self.pen)
        QPainter().drawRect(rectangle)
        # QPainter().begin()  # 开始绘制
        # QPainter().end()  # 结束绘制


# 实例化自定义信号源(全局)
global_mysignals = MySignals()

global_serial = serial_port_communication(None, None)

class CubeRobotUiDesigner(QMainWindow):
    def __init__(self):
        super(CubeRobotUiDesigner, self).__init__()
        # self.camera_id = {
        #     'up': 4,
        #     'down': 0,
        #     'front': 2,
        #     'back': 3
        # }
        # self.camera_id = {
        #     'up': 2,
        #     'down': 0,
        #     'front': 4,
        #     'back': 1
        # }
        self.camera_id = {
            'up': 2,
            'down': 0,
            'front': 4,
            'back': 1
        }
        # # 从文件中加载UI定义
        # qfile_CubeRobotUi = QFile(CubeRobotUi.ui)
        # qfile_CubeRobotUi.open(QFile.ReadOnly)
        # qfile_CubeRobotUi.close()
        # 动态加载ui文件
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        # self.ui = QUiLoader().load('C:/Users/35021/PychparmProjects/pythonProject/CubeRobotUi.ui')    # pyside2
        self.ui = uic.loadUi('C:/Users/35021/PycharmProjects/pythonProject/CubeRobotUi.ui')  # pyqt
        # self.ui = uic.loadUi('CubeRobotUi.ui')  # pyqt
        self.ui.VideoCaptureButton.clicked.connect(self.run)
        self.ui.SolveStartButton.clicked.connect(self.isstartsolve)

        self.lb = myLabel()
        # 需要在主线程中操作界面，自定义信号处理函数
        self.mysignals_init()
        # 串口UI初始化
        self.serialui_init()
        # 标志信号初始化
        self.flagsignals_init()
        # 测试按钮：夹爪；电机；气阀
        self.testbutton_init()
        # SVM模型训练UI初始化
        self.svmtapui_init()

    def svmtapui_init(self):
        self.filePathDisplay = self.ui.lineEdit
        self.openFileBtn = self.ui.pushButton
        self.openFileBtn.clicked.connect(self.openfile)

    def openfile(self):
        filepath, fileType = QFileDialog().getOpenFileName(None, "文件管理", os.getcwd(), "All Files(*);;Model Files(*.model)") # 返回当前所运行脚本的目录
        print(filepath, fileType)
        # QFileDialog.setFileMode(QFileDialog.AnyFile)    # 设置可以打开任何文件
        # QFileDialog.setFilter(QDir.Files)               # 文件过滤
        self.filePathDisplay.setText(filepath)
        # if QFileDialog().exec_():
        #     # 接受选中文件的路径，默认为列表
        #     filenames = QFileDialog().selectedFiles()
        #     print(filenames[0])

    def flagsignals_init(self):
        self.isStartSolve = False
        self.isExitThread = False

    def mysignals_init(self):
        global_mysignals.showImageSignal.connect(self.showImageToLabel)     # 图像显示
        global_mysignals.printSerialDataSignal.connect(self.printserialdata)    # 数据打印
        global_mysignals.displayRunTimeSignal.connect(self.displayruntime)

    def testbutton_init(self):
        self.mechanicalClawOpenL =self.ui.radioButton_9
        self.mechanicalClawCloseL = self.ui.radioButton_10
        self.mechanicalClawOpenR = self.ui.radioButton_7
        self.mechanicalClawCloseR = self.ui.radioButton_8
        self.motorRelease = self.ui.radioButton_3
        self.motorLock = self.ui.radioButton_4
        self.airValveOpen = self.ui.radioButton_11
        self.airValveClose = self.ui.radioButton_12

        self.btnlist = [self.mechanicalClawOpenL, self.mechanicalClawCloseL,
                   self.mechanicalClawOpenR, self.mechanicalClawCloseR,
                   self.motorRelease, self.motorLock,
                   self.airValveOpen, self.airValveClose]
        for btnobj in self.btnlist:
            btnobj.toggled.connect(lambda : self.testbuttoncontral(btnobj))

    def testbuttoncontral(self, btn):
        if btn == self.mechanicalClawOpenL:
            if btn.isChecked() == True:      # isChecked() 返回单选按钮的状态，返回值True或False
                self.mechanicalClawCloseL.setCheckanle(False)
        elif btn == self.mechanicalClawCloseL:
            if btn.isChecked() == True:
                self.mechanicalClawOpenL.setCheckanle(False)
        elif btn == self.mechanicalClawOpenR:
            if btn.isChecked() == True:      # isChecked() 返回单选按钮的状态，返回值True或False
                self.mechanicalClawCloseR.setCheckanle(False)
        elif btn == self.mechanicalClawCloseR:
            if btn.isChecked() == True:
                self.mechanicalClawOpenR.setCheckanle(False)
        elif btn == self.motorRelease:
            if btn.isChecked() == True:      # isChecked() 返回单选按钮的状态，返回值True或False
                self.motorLock.setCheckanle(False)
        elif btn == self.mechanicalClawCloseR:
            if btn.isChecked() == True:
                self.motorLock.setCheckanle(False)
        elif btn == self.airValveOpen:
            if btn.isChecked() == True:      # isChecked() 返回单选按钮的状态，返回值True或False
                self.airValveClose.setCheckanle(False)
        elif btn == self.airValveClose:
            if btn.isChecked() == True:
                self.airValveOpen.setCheckanle(False)
        # print("btn=", str(btn))

    def serialui_init(self):
        global global_serial
        self.serialCOMBox = self.ui.comboBox_2
        self.serialBaudrateBox = self.ui.comboBox_3
        self.serialbtn = self.ui.radioButton_2
        self.serialTextBrowser = self.ui.serialOutput_2


        self._serial = global_serial
        self.serialBaudrate = self.serialBaudrateBox.currentText()  # 默认波特率为115200
        self.list_ports = self._serial.list_serialports()
        try:
            self.serialCOM = self.list_ports[0]
        except:
            print("检测串口失败！")

        self.serialCOMBox.addItems(self.list_ports)  # 从列表中添加下拉选项
        self.serialCOMBox.currentIndexChanged.connect(self.serialCOMchange)  # 当下拉索引发生改变时发射信号触发绑定的事件
        self.serialBaudrateBox.currentIndexChanged.connect(self.serialBaudratechange)  # 当下拉索引发生改变时发射信号触发绑定的事件
        self.serialbtn.toggled.connect(self.buttonstate)  # 状态发生改变信号  调用槽函数

    def serialCOMchange(self):
        self.serialCOM = self.serialCOMBox.currentText()  # currentText()：返回选中选项的文本
        print(self.serialCOM)

    def serialBaudratechange(self):
        self.serialBaudrate = self.serialBaudrateBox.currentText()
        print(self.serialBaudrate)

    def buttonstate(self):
        if self.serialbtn.isChecked() == True:      # isChecked() 返回单选按钮的状态，返回值True或False
            try:
                print("打开串口")
                self._serial.__init__(self.serialCOM, self.serialBaudrate)
                self._serial.open_serial()
                self.serialbtn.setText("关闭串口")
            except:
                print("serial error!")
        else:
            self.serialbtn.setText("打开串口")
            print("关闭串口")

    def showImageToLabel(self, img, position, detectMode, camera_id):
        # position = ['front', 'back', 'up', 'down']
        # detectMode = {'one': 0, 'two': 1}  # 识别模式
        if detectMode == 0:  # 单个魔方面
            if position == 'front':
                displayObject = self.ui.labelDisplayFrontImage
            else:
                displayObject = self.ui.labelDisplayBackImage
        else:  # 两个魔方面
            if position == 'up':
                displayObject = self.ui.labelDisplayUpImage
            else:
                displayObject = self.ui.labelDisplayDownImage
        imgToNdarray = np.array(img, dtype='uint8')             # list对象转换成ndarray对象
        frame = cv2.cvtColor(imgToNdarray, cv2.COLOR_BGR2RGB)  # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
        image_height, image_width, image_depth = frame.shape  # 读取图像高宽深度
        Qframe = QImage(frame.data, image_width, image_height, image_width * image_depth, QImage.Format_RGB888)
        displayObject.setPixmap(QPixmap.fromImage(Qframe))  # 图片显示
        displayObject.setScaledContents(True)  # 图片自适应窗口大小
        # Image = cv2.resize(img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_CUBIC)    # 改变图像尺寸，默认图像大小为320*240
        # frame = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)                      # 转RGB
        # image_height, image_width, image_depth = frame.shape                # 读取图像高宽深度
        # Qframe = QImage(frame.data, image_width, image_height, image_width * image_depth, QImage.Format_RGB888)
        # displayObject.setPixmap(QPixmap.fromImage(Qframe))      # 图片显示
        # #self.ui.labelDisplayImage.setGeometry(30, 20, 640, 480)           # 设置窗口的位置和大小
        # displayObject.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # 左上角对齐
        # displayObject.setStyleSheet("color:red")
        # displayObject.setWindowTitle('photo')  # 设置窗口的标题

    def printserialdata(self, serialData):
        self.serialTextBrowser.setPlainText(serialData)

    def displayruntime(self, runTime):
        self.ui.doubleSpinBox.setValue(runTime[0])

    def start_thread(self):
        cubeSvm = CubeSvm()
        # 判断svm颜色模型是否训练完成
        # svm_color_model_isfinish = 1
        # if svm_color_model_isfinish:
        #     print('svm_color_model is OK')
        # else:
        #     CubeSvm().creat_svm_color_model()   # 创建svm颜色模型
        #     print('creat svm color model is finish')
        # 串口处理
        # _serial = serial_port_communication(None, 115200)
        # list_ports = _serial.list_serialports()
        # _serial.__init__(list_ports[0], 115200)
        # _serial.open_serial()
        # while True:
        #     _serial.serial_write_ascii("B U' L' D' R' D' L2 D' L F' L' D F2 R2 U R2 B2 U2 L2 F2 D'")
        # 多线程处理
        # camera_id = {
        #     'up': 2,
        #     'down': 0,
        #     'front': 4,
        #     'back': 1
        # }
        self.lineEditText = self.ui.lineEdit_2.text()
        self.camera_id = {
            'up': int(self.lineEditText[0]),
            'down': int(self.lineEditText[1]),
            'front': int(self.lineEditText[2]),
            'back': int(self.lineEditText[3])
        }
        print(self.camera_id)
        detection_mode = {'one': 0, 'two': 1}  # 识别模式
        # cap = cv2.VideoCapture(camera_id['down'])
        # _, Image = cap.read()
        # title = "OriginImage" + str(camera_id['down'])
        # cv2.imshow(title, Image)
        # cv2.waitKey(1)
        # plt.imshow(Image[:, :, ::-1])
        # # plt.show()
        self.thread1 = mythread(self.camera_id['down'], detection_mode['two'], 'down')
        self.thread2 = mythread(self.camera_id['up'], detection_mode['two'], 'up')
        self.thread3 = mythread(self.camera_id['front'], detection_mode['one'], 'front')
        self.thread4 = mythread(self.camera_id['back'], detection_mode['one'], 'back')
        # self.thread5 = mythread_solve_cube()

        self.thread1.start()  # 启动线程
        self.thread2.start()
        self.thread3.start()  # 启动线程
        self.thread4.start()


        # thread1.join()  # 等待至线程终止
        # thread2.join()
        # thread3.join()  # 等待至线程终止
        # thread4.join()
        # thread5.join()

    def isstartsolve(self):
        if self.ui.SolveStartButton.text() == "暂停识别":
            self.ui.SolveStartButton.setText("启动识别")
            self.isStartSolve = False
        else:
            if self.ui.VideoCaptureButton.text() == "关闭摄像头":
                self.ui.SolveStartButton.setText("暂停识别")
                self.isStartSolve = True
                self.thread5 = mythread_solve_cube()
                self.thread5.start()    # 开启解算线程

    def exit_thread(self):
        self.isExitThread = True

    def run(self):
        if self.ui.VideoCaptureButton.text() == "关闭摄像头":
            self.exit_thread()
            self.ui.VideoCaptureButton.setText("开启摄像头")
            self.ui.SolveStartButton.setText("启动识别")
        else:
            self.flagsignals_init()
            tic = timer()
            self.start_thread()
            toc = timer()
            runtime = toc - tic
            print("time:", runtime)  # 输出的时间，秒为单位
            global_mysignals.displayRunTimeSignal.emit([runtime])
            self.ui.VideoCaptureButton.setText("关闭摄像头")

# 重写线程

if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 提供了整个图形界面程序的底层管理功能，所以必须在任何界面控件对象创建前，先创建它。
    _CubeRobotUiDesigner = CubeRobotUiDesigner()
    _CubeRobotUiDesigner.ui.show()  # 放在主窗口的控件，要能全部显示在界面上
    sys.exit(app.exec_())  # 进入QApplication的事件处理循环，接收用户的输入事件（），并且分配给相应的对象去处理。
