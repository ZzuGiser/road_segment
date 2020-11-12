# road_segment
项目中利用mask rcnn实现矢量数据几何精度和数据质量的提升

### 环境配置
```bash
conda install --yes --file requirements.txt
```

### 主要功能介绍
sample_create_main.py 利用shp和tif数据自动化裁剪数据，制作训练样本集
road_train 利用样本集进行训练
raod_veriy 对训练结果进行验证
road_accuracy 利用训练模型，对shp和tif数据进行同名点结果验证