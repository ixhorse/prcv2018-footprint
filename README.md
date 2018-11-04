# PRCV2018 大连恒锐足迹图像人身识别和信息挖掘竞赛
### 环境要求:
    python=3.6
        pytorch=0.4.1
        numpy
        pandas
        scikit-learn
        matplotlib
        pillow
        opencv-python

### 代码说明：
    main.py         主程序
    config.ini      配置文件
    train.py        训练函数
    predict.py      测试函数
    utils.py        工具函数
    condensenet.py  模型代码
    layers.py       网络层函数
    aug.py          预处理函数
    split.py        训练集测试集分割函数
    /output         模型输出以及预测输出目录
    /dataset        处理后数据目录

### 配置文件说明：
    RefPath:            训练集目录，要求训练目录下包含train.txt
    TestPath:           测试集目录
    Task:               1为识别任务，2为挖掘任务
    validation:         是否训练时验证，复现结果的话不用更改
    pretrain:           是否使用训练好的模型，为True时使用模型文件进行预测，为False时重新训练网络
    ModelPath_type1:    当pretrain=True时，需设置模型1的路径
    ModelPath_type2:    当pretrain=True时，需设置模型2的路径

## 运行：
设置好配置文件后，运行python main.py
结果输出在output目录下，识别任务结果为result_task1.txt，挖掘任务结果为result_task2.txt