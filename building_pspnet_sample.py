#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：road_segment -> building_clip
@IDE    ：PyCharm
@Author ：shaoxin
@Date   ：2020/12/17 14:53
@Desc   ：
=================================================='''
import datetime
import logging
import os
import gdal
import numpy as np
from PIL import Image

TIFPATH = "./tif_and_shp/CJ2.tif"
SHPPATH = "./tif_and_shp/image_building/guangzhou_mask.tif"
SAVEPATH = "./seg_data"
CROPSIZE = 256
REPETITIONRATE = 0.1
ALL_IMAGE_NUM = 100


class TifSample(object):
    def __init__(self, TifPath=TIFPATH, ShpPath=SHPPATH, SavePath=SAVEPATH, CropSize=CROPSIZE,
                 RepetitionRate=REPETITIONRATE):
        self.tif_path = TifPath
        self.shp_path = ShpPath
        self.sava_path = SavePath
        self.crop_size = CropSize
        self.repetition_rate = RepetitionRate
        self.img_num = 0
        self.train_txt = []

    #  读取tif数据集
    def readTif(self, fileName):
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
        return dataset

    #  保存tif文件函数
    def writeTiff(self, im_data, path):

        if len(im_data.shape) == 2:
            im = np.concatenate(
                (im_data[:, :, np.newaxis], im_data[:, :, np.newaxis], im_data[:, :, np.newaxis]), axis=2)
            im = Image.fromarray(im)
        else:
            im = np.concatenate(
                (im_data[0, :, :, np.newaxis], im_data[1, :, :, np.newaxis], im_data[2, :, :, np.newaxis]), axis=2)
            im = im.astype(np.uint8)
            im = Image.fromarray(im).convert('RGB')
        im.save(path)

    '''
    滑动窗口裁剪函数
    TifPath 影像路径
    SavePath 裁剪后保存目录
    CropSize 裁剪尺寸
    RepetitionRate 重复率
    '''

    def TifCrop(self):
        CropSize = self.crop_size
        RepetitionRate = self.repetition_rate
        dataset_img = self.readTif(self.tif_path)
        dataset_shp = self.readTif(self.shp_path)
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
        img_shp = dataset_shp.ReadAsArray(0, 0, width, height)

        for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            if self.img_num > ALL_IMAGE_NUM:
                break
            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                #  如果图像是单波段
                if self.img_num > ALL_IMAGE_NUM:
                    break
                if (len(img.shape) == 2):
                    cropped = img[
                              int(i * CropSize * (1 - RepetitionRate)): int(
                                  i * CropSize * (1 - RepetitionRate)) + CropSize,
                              int(j * CropSize * (1 - RepetitionRate)): int(
                                  j * CropSize * (1 - RepetitionRate)) + CropSize]
                #  如果图像是多波段
                else:
                    cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)): int(
                                  i * CropSize * (1 - RepetitionRate)) + CropSize,
                              int(j * CropSize * (1 - RepetitionRate)): int(
                                  j * CropSize * (1 - RepetitionRate)) + CropSize]
                #  写图像
                cropped_shp = img_shp[
                              int(i * CropSize * (1 - RepetitionRate)): int(
                                  i * CropSize * (1 - RepetitionRate)) + CropSize,
                              int(j * CropSize * (1 - RepetitionRate)): int(
                                  j * CropSize * (1 - RepetitionRate)) + CropSize]
                color = int(np.sum(cropped) / (CropSize * CropSize))
                if color < 70: continue
                img_name = "{:d}_{:d}_{:d}.jpg".format(self.img_num, j + CropSize // 2, i + CropSize // 2)
                shp_name = "{:d}_{:d}_{:d}.png".format(self.img_num, j + CropSize // 2, i + CropSize // 2)
                self.writeTiff(cropped, os.path.join(self.sava_path, "jpg", img_name))
                self.writeTiff(cropped_shp, os.path.join(self.sava_path, "png", shp_name))
                #  文件名 + 1
                self.train_txt.append("{};{}\n".format(img_name, shp_name))
                logging.info("write tif:{}".format(img_name))
                self.img_num = self.img_num + 1
        train_path = os.path.join(self.sava_path, "train.txt")
        with open(train_path, "w") as f:
            for train_line in self.train_txt:
                f.writelines(train_line)


if __name__ == '__main__':
    output_pack = '{:%Y%m%d_%H%M}_road_accuracy'.format(datetime.datetime.now())
    output_path = os.path.join(SAVEPATH, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'jpg'))
        os.makedirs(os.path.join(output_path, 'png'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')

    tif_path = ""
    sava_path = ""

    #  将影像1裁剪为重复率为0.1的256×256的数据集
    tif_sample = TifSample(SavePath=output_path)
    tif_sample.TifCrop()
    # TifCrop(r"Data\data2\label\label.tif",
    #         r"data\train\label1", 256, 0.1)
