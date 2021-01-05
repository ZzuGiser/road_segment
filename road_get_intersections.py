#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：road_segment -> road_get_intersections
@IDE    ：PyCharm
@Author ：shaoxin
@Date   ：2021/1/3 22:49
@Desc   ：
=================================================='''

import geopandas
from geopandas.tools import sjoin
import os
import numpy as np
import datetime
import logging
import geopandas as gpd
import sys
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr
# point = geopandas.GeoDataFrame.from_file('points.shp')
# poly  = geopandas.GeoDataFrame.from_file('multipol.shp')
# pointInPolys = sjoin(point, poly, how='left')
# grouped = pointInPolys.groupby('index_right')
# list(grouped)
SHP_PATH = './tif_and_shp/line_shp/nanchang.shp'
OUT_PACK = './result'
OUT_NAME = 'intersection'
OUT_PATH = os.path.join(OUT_PACK, OUT_NAME)
CLUSTER_LEN = 0.00100
LAYER_NAME = 'point_intersection'
ALL_NUM = sys.maxsize


class GetIntersection(object):
    def __init__(self, shp_path=SHP_PATH, outpath=OUT_PATH):
        self.shp_data = gpd.read_file(shp_path)
        self.out_name = outpath

        self.cluster_len = CLUSTER_LEN
        self.layer_name = LAYER_NAME

        self.all_res_path = os.path.join(outpath, "a_intersection_lat_lon.csv")
        self.culster_csv = os.path.join(outpath, "a_intersection_lat_lon_culster.csv")
        self.cluster_point_path = os.path.join(outpath, 'a_intersection_lat_lon_culster_point.csv')
        self.culster_png = os.path.join(outpath, "a_Clustering.png")
        self.culster_res_png = os.path.join(outpath, "a_Clustering_res.png")

    def get_intersection(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
        strVectorFile = self.out_name  # 定义写入路径及文件名
        ogr.RegisterAll()  # 注册所有的驱动
        strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s 驱动不可用！\n", strDriverName)
        oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
        if oDS == None:
            print("创建文件【%s】失败！", strVectorFile)
        srs = osr.SpatialReference()  # 创建空间参考
        srs.ImportFromEPSG(4326)  # 定义地理坐标系WGS1984
        papszLCO = []
        # 创建图层，创建一个多边形图层,"TestPolygon"->属性表名
        oLayer = oDS.CreateLayer(self.layer_name, srs, ogr.wkbPoint, papszLCO)
        if oLayer == None:
            print("图层创建失败！\n")
        oLayer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
        # 创建 名为Latitude 的字段数据类型为 Double 23/15 . ogr.OFTReal->shapefile Double
        oLayer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))
        oLayer.CreateField(ogr.FieldDefn("shp_i_j", ogr.OFTString))
        self.handle_intersection(oLayer, oDS)

    def handle_intersection(self, oLayer, oDS):
        logging.info("start handle intersection")
        geometrys = self.shp_data.geometry
        intersection_res = []
        for shp_i, geo_i in enumerate(geometrys[:-1]):
            logging.info("_______handle {:d} of {:d}_______".format(shp_i, len(geometrys)))
            if shp_i > ALL_NUM:
                break
            for shp_j, geo_j in enumerate(geometrys[shp_i + 1:]):
                if not geo_i.intersects(geo_j):
                    continue
                line_i = [(geo_i.xy[0][k], geo_i.xy[1][k]) for k in range(len(geo_i.xy[0]))]
                line_j = [(geo_j.xy[0][k], geo_j.xy[1][k]) for k in range(len(geo_j.xy[0]))]
                intersection_point = self.poly_intersection(line_i, line_j)
                if intersection_point:
                    lat, lon, shp_i_j = intersection_point[0], intersection_point[1], "{:d}_{:d}".format(shp_i, shp_j)
                    intersection_res.append([lat, lon, shp_i_j])

        res_data_frame = pd.DataFrame(intersection_res, columns=['lat', 'lon', 'shp_i_j'])
        res_data_frame.to_csv(self.all_res_path)
        cluster_res = self.culster(res_data_frame[['lat', 'lon']])
        cluster_res.to_csv(self.culster_csv)

        cluster_point_out = []
        for i in range(cluster_res['jllable'].max()):
            class_i = cluster_res[cluster_res['jllable'] == i]
            lat, lon = [class_i['lat'].mean(), class_i['lon'].mean()]
            cluster_point_out.append([lat, lon])
            feature = ogr.Feature(oLayer.GetLayerDefn())
            # 和设置字段内容进行关联  ,从数据源中写入数据
            feature.SetField("Latitude", lat)
            feature.SetField("Longitude", lon)
            feature.SetField("shp_i_j", str(i))
            # 创建WKT 文本点
            wkt = "POINT(%f %f)" % (
                float(lat), float(lon))
            # 生成实体点
            point = ogr.CreateGeometryFromWkt(wkt)
            # 使用点
            feature.SetGeometry(point)
            # 添加点
            oLayer.CreateFeature(feature)

        cluster_point_out_frame = pd.DataFrame(cluster_point_out, columns=['lat', 'lon'])
        cluster_point_out_frame.to_csv(self.cluster_point_path)
        logging.info("end handle intersection")
        oDS.Destroy()

    def culster(self, cluster_data):
        res_dbscan = DBSCAN(eps=self.cluster_len, min_samples=1).fit(
            cluster_data)  # eps： DBSCAN算法参数，即我们的𝜖ϵ-邻域的距离阈值，和样本距离超过𝜖ϵ的样本点不在𝜖ϵ-邻域内。
        cluster_data['jllable'] = res_dbscan.labels_
        ##可视化
        # plt.cla()
        # d = cluster_data[cluster_data['jllable'] == 0]
        # plt.plot(d['lat'], d['lon'], 'r.')
        # d = cluster_data[cluster_data['jllable'] == -1]
        # plt.plot(d['lat'], d['lon'], 'go')
        # plt.gcf().savefig(self.culster_png)

        # plt.show()
        return cluster_data

    def get_cross_angle(self, l1, l2):
        arr_a = np.array([(l1[1][0] - l1[0][0]), (l1[1][1] - l1[0][1])])  # 向量a
        arr_b = np.array([(l2[1][0] - l2[0][0]), (l2[1][1] - l2[0][1])])  # 向量b
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
        return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度

    def line_intersection(self, line1, line2):
        def cross(p1, p2, p3):  # 跨立实验
            x1 = p2[0] - p1[0]
            y1 = p2[1] - p1[1]
            x2 = p3[0] - p1[0]
            y2 = p3[1] - p1[1]
            return x1 * y2 - x2 * y1

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        div = det(xdiff, ydiff)
        if div == 0:
            return None
        # 若通过快速排斥则进行跨立实验
        if (cross(line1[0], line1[1], line2[0]) * cross(line1[0], line1[1], line2[1]) <= 0
            and cross(line2[0], line2[1], line1[0]) * cross(line2[0], line2[1], line1[1]) <= 0) \
                and self.get_cross_angle(line1, line2) > (60 / np.pi):
            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y
        else:
            return None

    def poly_intersection(self, poly1, poly2):

        for i, p1_first_point in enumerate(poly1[:-1]):
            p1_second_point = poly1[i + 1]

            for j, p2_first_point in enumerate(poly2[:-1]):
                p2_second_point = poly2[j + 1]
                res = self.line_intersection((p1_first_point, p1_second_point), (p2_first_point, p2_second_point))

                if res:
                    return res
        return False


if __name__ == '__main__':
    output_pack = '{:%Y%m%d_%H%M}_intersection_to_shp'.format(datetime.datetime.now())
    output_path = os.path.join(OUT_PACK, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO)
    intersection_generate = GetIntersection(outpath=output_path)
    intersection_generate.get_intersection()
    # PL1 = ((-1, -1), (1, -1), (1, 2))
    # PL2 = ((0, 1), (2, 1))
    # print(poly_intersection(PL1, PL2))
