#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
from PIL import Image

import os
import numpy as np
import datetime
import logging
import mrcnn.model as modellib
from mrcnn import visualize
import building_train
from building_sample_create_main import TIF_TRANS
import building_sample_create_main
import geopandas as gpd
import sys
import io
import cv2
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
# Import COCO config
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_building_0020.h5")


# Download COCO trained weights from Releases if needed

class InferenceConfig(building_train.BuildingConfig):
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
CLASS_NAMES = ['not_defined', 'building']
OUTPUT_PATH = os.path.join(ROOT_DIR, 'result')

TIF_PATH = building_sample_create_main.TIF_PATH
SHP_PATH = building_sample_create_main.SHP_PATH
CROP_SIZE = 400
NEW_SHP_NAME = 'building_detect.shp'
ALL_NUM = 10000


class Remote2Shp(object):
    def __init__(self, tif_path=TIF_PATH, shp_path=SHP_PATH, new_shp_path=NEW_SHP_NAME, model=MODEL,
                 class_names=CLASS_NAMES,
                 output_path=OUTPUT_PATH):
        self.model = model
        self.class_names = class_names
        self.ouput_path = output_path
        self.tif_path = tif_path
        self.tif_img = gdal.Open(tif_path)
        self.shp_data = gpd.read_file(shp_path)
        self.image_num = 0
        self.shp_img = np.ones((self.tif_img.RasterXSize, self.tif_img.RasterYSize)) * 255
        self.new_shp_path = os.path.join(output_path, new_shp_path)
        self.oDS = self.init_create_shp()

    def init_create_shp(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
        strVectorFile = self.new_shp_path  # 定义写入路径及文件名
        ogr.RegisterAll()  # 注册所有的驱动
        strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s 驱动不可用！\n", strDriverName)
        oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
        if oDS == None:
            print("创建文件【%s】失败！", strVectorFile)
        return oDS

    def creaate_val_sample(self, crop_size=CROP_SIZE):
        srs = osr.SpatialReference()  # 创建空间参考
        srs.ImportFromEPSG(4326)  # 定义地理坐标系WGS1984
        papszLCO = []
        # 创建图层，创建一个多边形图层,"TestPolygon"->属性表名
        oLayer = self.oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("图层创建失败！\n")
        '''下面添加矢量数据，属性表数据、矢量数据坐标'''
        oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
        oLayer.CreateField(oFieldID, 1)
        oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)  # 创建一个叫FieldName的字符型属性
        oFieldName.SetWidth(100)  # 定义字符长度为100
        oLayer.CreateField(oFieldName, 1)
        tif_tran = TIF_TRANS(self.tif_path)
        for shp_i, geo in enumerate(self.shp_data.geometry):
            if shp_i > ALL_NUM:
                break
            row, col = tif_tran.geo2imagexy(geo.centroid.x, geo.centroid.y)
            x_df, y_df = int(crop_size / 2), int(crop_size / 2)
            raster_crop = self.tif_crop(crop_size, row, col, x_df, y_df)
            if len(raster_crop) == 0:
                continue
            image = raster_crop
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.concatenate((image, image, image), axis=2)
            else:
                image = np.stack(image, axis=2)
            w, h, _ = image.shape  # w = 400,h = 400
            results = model.detect([image], verbose=1)
            # Visualize results
            r = results[0]
            visualize.add_instances(r['rois'], r['masks'], r['class_ids'], oLayer, [row, col], tif_tran,shp_i)
            # img_out_path = os.path.join(self.ouput_path,"{}.jpg".format(str(shp_i)))
            # image = image.astype(np.uint8)
            # image = Image.fromarray(image).convert('RGB')
            # image.save(img_out_path)
        self.oDS.Destroy()
        print("数据集创建完成！\n")

    def tif_crop(self, crop_size, x, y, x_df, y_df):
        dataset_img = self.tif_img
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

        #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
        new_name = '{}_{}_{}.jpg'.format(self.image_num, int(x), int(y))
        #  裁剪图片,重复率为RepetitionRate
        x_min, x_max = x - x_df, x + crop_size - x_df
        y_min, y_max = y - y_df, y + crop_size - y_df

        if (len(img.shape) == 2):
            cropped = img[int(y_min): int(y_max), int(x_min): int(x_max)]
        # 如果图像是多波段
        else:
            if img.shape[0] > 3:
                cropped = img[0:3, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
            else:
                cropped = img[:, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
        # 写图像
        if x_min < 0 or x_max > height or y_min < 0 or y_max > width:
            return []

        self.image_num += 1
        logging.info('crop image name:{}'.format(new_name))
        return cropped

    def remote2Shp(self):
        self.creaate_val_sample()


if __name__ == '__main__':

    model = MODEL
    class_names = CLASS_NAMES
    output_pack = '{:%Y%m%d_%H%M}_building_to_shp'.format(datetime.datetime.now())
    output_path = os.path.join(OUTPUT_PATH, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')
    remote2shp = Remote2Shp(output_path=output_path)
    remote2shp.remote2Shp()
