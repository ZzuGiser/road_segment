#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : road_accuracy.py
# @Author: shao
# @Date  : 2020/11/10
# @Desc  :

import geopandas as gpd
from matplotlib import pyplot as plt
# from osgeo import gdal
# from osgeo import osr
import numpy as np
import os
import gdal
import json
from PIL import Image

import math
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import datetime
import re
from sklearn.cluster import DBSCAN
import pandas as pd
import logging
import mrcnn.model as modellib
from mrcnn import visualize
import road_train
from road_sample_create_main import TIF_TRANS

CUR_PATH = r'./'
TIF_PATH = os.path.join(CUR_PATH, r'tif_and_shp/CJ2.tif')
SHP_PATH = os.path.join(CUR_PATH, r'tif_and_shp/point/Correction.shp')
CROP_SIZE = 400
OUTPUT_PATH = os.path.join(CUR_PATH, 'result')

ROOT_DIR = os.path.abspath(CUR_PATH)
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
# Import COCO config
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_road_0020_copy.h5")
DIS_THRESHOLD = 50


# Download COCO trained weights from Releases if needed
class InferenceConfig(road_train.RoadConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()
# Create model object in inference mode.
MODEL = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
MODEL.load_weights(COCO_MODEL_PATH, by_name=True)
CLASS_NAMES = ['not_defined', 'Road']


class Road_Accuracy(object):
    def __init__(self, tif_path=TIF_PATH, shp_path=SHP_PATH, output_path=OUTPUT_PATH, model=MODEL,
                 class_names=CLASS_NAMES):
        gdal.AllRegister()
        self.tif_path = tif_path
        self.dataset = self.readTif(tif_path)
        self.shp_path = shp_path
        self.shp_data = gpd.read_file(shp_path)
        self.save_path = output_path
        self.model = model
        self.class_names = class_names
        self.all_patch_res_path = os.path.join(output_path, 'a_all_patch_res.csv')
        self.filter_patch_res_path = os.path.join(output_path, 'a_filter_patch_res.csv')
        self.culster_png = os.path.join(output_path, 'a_Clustering.png')
        self.culster_csv = os.path.join(output_path, 'a_Clustering.csv')
        self.img_num = 1

    # def geo2lonlat(self, x, y):
    #     '''
    #     将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    #     :param dataset: GDAL地理数据
    #     :param x: 投影坐标x
    #     :param y: 投影坐标y
    #     :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    #     '''
    #     prosrs, geosrs = getSRSPair(self.dataset)
    #     ct = osr.CoordinateTransformation(prosrs, geosrs)
    #     coords = ct.TransformPoint(x, y)
    #     return coords[:2]

    def imagexy2geo(self, row, col):
        '''
            根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        '''
        trans = self.dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

    #  读取tif数据集
    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
        return dataset

        #  保存tif文件函数

    def tif_crop(self, crop_size, x, y, x_df, y_df):
        dataset_img = self.dataset
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

        #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
        new_name = '{}_{}_{}.jpg'.format(self.img_num, int(x), int(y))
        #  裁剪图片,重复率为RepetitionRate
        x_min, x_max = x - x_df, x + crop_size - x_df
        y_min, y_max = y - y_df, y + crop_size - y_df

        if (len(img.shape) == 2):
            cropped = img[int(y_min): int(y_max), int(x_min): int(x_max)]
        # 如果图像是多波段
        else:
            cropped = img[:, int(y_min): int(y_max),
                      int(x_min): int(x_max)]
        # 写图像
        try:
            logging.info('crop image name:{}'.format(new_name))
            self.img_num += 1
            return cropped, new_name
        except:
            return None, None

    def get_accuracy(self, crop_size=CROP_SIZE):
        res = []
        for geo in self.shp_data.geometry:
            lon, lat = geo.x, geo.y
            row, col = self.geo2imagexy(lon, lat)
            x_df, y_df = int(crop_size / 2), int(crop_size / 2)
            raster, raster_name = self.tif_crop(crop_size, row, col, x_df, y_df)
            if raster_name == None: continue
            raster = raster[:, :, np.newaxis]
            raster = np.concatenate((raster, raster, raster), axis=2)
            self.do_detech_roads(raster, raster_name, res)
        res_data_frame = pd.DataFrame(res, columns=['offset_x', 'offset_y', 'dis', 'x_before', 'y_before', 'x_after',
                                                    'y_after', 'is_reals', 'img_path'])
        all_patch_res = res_data_frame[res_data_frame['dis'] != 0]
        all_patch_res.to_csv(self.all_patch_res_path)
        cluster_res = self.culster(all_patch_res[['offset_x', 'offset_y']])
        cluster_res.to_csv(self.culster_csv)
        filter_patch_res = all_patch_res[cluster_res['jllable'] == 0]
        filter_patch_res.to_csv(self.filter_patch_res_path)
        accuracy = len(all_patch_res[all_patch_res['is_reals']]) / float(len(all_patch_res) + 0.1)
        filter_accuracy = len(filter_patch_res[filter_patch_res['is_reals']]) / float(len(filter_patch_res) + 0.1)
        return accuracy, filter_accuracy

    def center_point(self, points, w, h):
        '''center_point 计算提取坐标的实际坐标的差值 '''
        try:
            dis_list = []
            offset = []
            is_reals = {}
            for xy in points:
                y1, x1, y2, x2 = xy
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                off_setp = [x - h / 2, y - w / 2]
                offset.append(off_setp)
                dis = math.sqrt(math.pow(x - h / 2, 2) + math.pow(y - w / 2, 2))
                is_real = dis < DIS_THRESHOLD
                is_reals.setdefault(len(dis_list), is_real)
                dis_list.append(dis)
            min_index = dis_list.index(min(dis_list))  # 最大值的索引
            return dis_list[min_index], offset[min_index], is_reals.get(min_index)
        except:
            return 0, [400, 400], False

    def do_detech_roads(self, image, image_name, res):
        ouput_path = self.save_path
        img_path = os.path.join(ouput_path, image_name)
        w, h, _ = image.shape  # w = 400,h = 400
        results = self.model.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'],
                                 self.class_names, r['scores'], save_name=image_name, save_path=ouput_path)
        dis, offset_xy, is_real = self.center_point(r['rois'], w, h)
        m = re.match(r'(\d+)_(\d+)_(\d+).jpg', image_name)
        row_point, col_point = int(m.group(2)), int(m.group(3))
        x_before, y_before = self.imagexy2geo(col_point, row_point)
        x_after, y_after = self.imagexy2geo(col_point + offset_xy[1], row_point + offset_xy[0])
        temp = [offset_xy[0], offset_xy[1], dis, x_before, y_before, x_after, y_after, is_real, img_path]
        res.append(temp)
        temp_str = [str(val) for val in temp]
        logging.info('_'.join(temp_str))

    def culster(self, cluster_data):
        res_dbscan = DBSCAN(eps=20, min_samples=5).fit(
            cluster_data)  # eps： DBSCAN算法参数，即我们的𝜖ϵ-邻域的距离阈值，和样本距离超过𝜖ϵ的样本点不在𝜖ϵ-邻域内。
        cluster_data['jllable'] = res_dbscan.labels_
        ##可视化
        plt.cla()
        d = cluster_data[cluster_data['jllable'] == 0]
        plt.plot(d['offset_x'], d['offset_y'], 'r.')
        d = cluster_data[cluster_data['jllable'] == -1]
        plt.plot(d['offset_x'], d['offset_y'], 'go')
        plt.gcf().savefig(self.culster_png)
        # plt.show()
        return cluster_data


if __name__ == '__main__':
    tif_path = TIF_PATH
    shp_path = SHP_PATH
    model = MODEL
    class_names = CLASS_NAMES
    output_pack = '{:%Y%m%d_%H%M}_road_accuracy'.format(datetime.datetime.now())
    output_path = os.path.join(OUTPUT_PATH, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')
    road_accuracy = Road_Accuracy(tif_path=tif_path, shp_path=shp_path, output_path=output_path, model=model,
                                  class_names=class_names)
    crop_size = CROP_SIZE
    accuracy, filter_accuracy = road_accuracy.get_accuracy(crop_size=crop_size)
    print('accuracy:{},filter_accuracy:{}'.format(accuracy, filter_accuracy))

    # all_path = r'D:\360download\code_targetdetection\road_sample\result\20201110_1650_road_accuracy\a_all_patch_res.csv'
    # filter_apth = r'D:\360download\code_targetdetection\road_sample\result\20201110_1650_road_accuracy\a_filter_patch_res.csv'
    # all_patch_res = pd.read_csv(all_path)
    # filter_patch_res = pd.read_csv(filter_apth)
    # accuracy = len(all_patch_res[all_patch_res['is_real']]) / float(len(all_patch_res) + 0.1)
    # filter_accuracy = len(filter_patch_res[filter_patch_res['is_real']]) / float(len(filter_patch_res) + 0.1)
    # print('accuracy:{},filter_accuracy:{}'.format(accuracy, filter_accuracy))
