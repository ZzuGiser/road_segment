#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：road_segment -> gui_main
@IDE    ：PyCharm
@Author ：shaoxin
@Date   ：2020/12/23 15:15
@Desc   ：
=================================================='''
import logging
import os
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import road_sample_create_main as road_sample
from tkinter.messagebox import showinfo
import datetime

from building_sample_create_main import TIF_HANDLE, SHP_HANDLE
from road_accuracy import Road_Accuracy

road_shp = ""
road_img = ""
road_output = ""
building_shp = ""
building_img = ""
building_output = ""


def road_fileopen():
    road_shp = askopenfilename()
    if road_shp:
        road_v1.set(road_shp)


def road_fileopen_1():
    road_img = askopenfilename()
    if road_img:
        road_v2.set(road_img)


def road_fileopen_2():
    road_output = askdirectory()
    if road_output:
        road_v3.set(road_output)


def road_get_sample():
    output_pack = '{:%Y%m%d_%H%M}_road_sample'.format(datetime.datetime.now())
    output_path = os.path.join(road_output, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_path = road_img
    shp_path = road_shp
    tif_handle = road_sample.TIF_HANDLE(path=img_path, save_path=output_path)
    shp_handle = road_sample.SHP_HANDLE(shp_path=shp_path)
    shp_handle.creaate_train_sample(tif_handle=tif_handle)

def road_get_accuracy():
    tif_path = road_img
    shp_path = road_shp
    output_pack = '{:%Y%m%d_%H%M}_road_accuracy'.format(datetime.datetime.now())
    output_path = os.path.join(road_output, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')
    road_accuracy = Road_Accuracy(tif_path=tif_path, shp_path=shp_path, output_path=output_path)
    accuracy, filter_accuracy = road_accuracy.get_accuracy()
    print('accuracy:{},filter_accuracy:{}'.format(accuracy, filter_accuracy))


def building_fileopen():
    building_shp = askopenfilename()
    if building_shp:
        building_v1.set(building_shp)


def building_fileopen_1():
    building_img = askopenfilename()
    if building_img:
        building_v2.set(building_img)


def building_fileopen_2():
    building_output = askdirectory()
    if building_output:
        building_v3.set(building_output)


def building_get_sample():
    tif_path = road_img
    shp_path = road_shp
    output_pack = '{:%Y%m%d_%H%M}_road_accuracy'.format(datetime.datetime.now())
    output_path = os.path.join(road_output, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tif_handle = TIF_HANDLE(path=tif_path, save_path=output_path)
    shp_handle = SHP_HANDLE(shp_path=shp_path)
    shp_handle.creaate_train_sample(tif_handle=tif_handle)


frameT = Tk()
frameT.geometry('650x500+400+200')
frameT.title('数据测试主界面')

# 定义第一个容器
frame_left = LabelFrame(frameT, text="道路交叉口测试项", labelanchor="n")
frame_left.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.4)

frame = Frame(frame_left)
frame.pack(padx=10, pady=10)  # 设置外边距
frame_1 = Frame(frame_left)
frame_1.pack(padx=10, pady=10)  # 设置外边距
frame_2 = Frame(frame_left)
frame_2.pack(padx=10, pady=10)
frame1 = Frame(frame_left)
frame1.pack(padx=10, pady=10)

road_v1 = StringVar()
road_v2 = StringVar()
road_v3 = StringVar()
ent = Entry(frame, width=50, textvariable=road_v1).pack(fill=X, side=LEFT)  # x方向填充,靠左
ent = Entry(frame_1, width=50, textvariable=road_v2).pack(fill=X, side=LEFT)  # x方向填充,靠左
ent = Entry(frame_2, width=50, textvariable=road_v3).pack(fill=X, side=LEFT)  # x方向填充,靠左

btn = Button(frame, width=20, text='矢量文件', font=("宋体", 10), command=road_fileopen).pack(fil=X, padx=10)
btn_1 = Button(frame_1, width=20, text='影像文件', font=("宋体", 10), command=road_fileopen_1).pack(fil=X, padx=10)
btn_2 = Button(frame_2, width=20, text='输出', font=("宋体", 10), command=road_fileopen_2).pack(fil=X, padx=10)

ext = Button(frame1, width=10, text='样本制作', font=("宋体", 10), command=road_get_sample).pack(fill=X, side=LEFT)
etb = Button(frame1, width=10, text='精度检测', font=("宋体", 10), command=road_get_accuracy).pack(fill=Y, padx=10)

# 定义第二个容器
frame_right = LabelFrame(frameT, text="建筑物测试项", labelanchor="n")
frame_right.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.4)

frame = Frame(frame_right)
frame.pack(padx=10, pady=10)  # 设置外边距
frame_1 = Frame(frame_right)
frame_1.pack(padx=10, pady=10)  # 设置外边距
frame_2 = Frame(frame_right)
frame_2.pack(padx=10, pady=10)
frame1 = Frame(frame_right)
frame1.pack(padx=10, pady=10)

building_v1 = StringVar()
building_v2 = StringVar()
building_v3 = StringVar()
ent = Entry(frame, width=50, textvariable=building_v1).pack(fill=X, side=LEFT)  # x方向填充,靠左
ent = Entry(frame_1, width=50, textvariable=building_v2).pack(fill=X, side=LEFT)  # x方向填充,靠左
ent = Entry(frame_2, width=50, textvariable=building_v3).pack(fill=X, side=LEFT)  # x方向填充,靠左

btn = Button(frame, width=20, text='矢量文件', font=("宋体", 10), command=building_fileopen).pack(fil=X, padx=10)
btn_1 = Button(frame_1, width=20, text='影像文件', font=("宋体", 10), command=building_fileopen_1).pack(fil=X, padx=10)
btn_2 = Button(frame_2, width=20, text='输出', font=("宋体", 10), command=building_fileopen_2).pack(fil=X, padx=10)

ext = Button(frame1, width=10, text='样本制作', font=("宋体", 10), command=match).pack(fill=X, side=LEFT)

frameT.mainloop()
