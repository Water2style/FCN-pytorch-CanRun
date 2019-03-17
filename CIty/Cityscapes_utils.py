# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import random
import os
import imageio

#############################
    # global variables #
#############################
root_dir  = "/home/water/DATA/CityScapes"
#创建相应路径
label_dir = os.path.join(root_dir, "gtFine") 
train_dir = os.path.join(label_dir, "train") #/home/water/DATA/cityscapes/gtFine/train
val_dir   = os.path.join(label_dir, "val")
test_dir  = os.path.join(label_dir, "test")
#各自创建标签图路径

# create dir for label index
#创建灰度图路径
label_idx_dir = os.path.join(root_dir, "Labeled_idx")  #/home/water/DATA/cityscapes/Labeled_idx
train_idx_dir = os.path.join(label_idx_dir, "train")#/home/water/DATA/cityscapes/Labeled_idx/train
val_idx_dir   = os.path.join(label_idx_dir, "val")
test_idx_dir  = os.path.join(label_idx_dir, "test")
#没有文件夹就创建
for dir in [train_idx_dir, val_idx_dir, test_idx_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
#然后现在就有了 /home/water/DATA/cityscapes/Labeled_idx/train等三个文件夹

train_file = os.path.join(root_dir, "train.csv")  #待会儿在根目录制造3个.csv存放数据路径
val_file   = os.path.join(root_dir, "val.csv")    #
test_file  = os.path.join(root_dir, "test.csv")   #

color2index = {}   #只定义了一个这样的“转换”字典
#index2color = {}不要吗！！！5555 不会可视化
                   


##去jupyter试试
#对于namedtuple，你不必再通过索引值进行访问，你可以把它看做一个字典通过名字进行访问，只不过其中的值是不能改变的。
#参数(元组大名，[，，，，，‘元组内容’])
Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'trainId', 
                   'category', 
                   'categoryId', 
                   'hasInstances', 
                   'ignoreInEval', 
                   'color'])


#可以根据这个学习一下，什么是忽视标签！！！！！！！！留下我们想要的！！！！！！！！
#trainID 改为255，ignoreInEval改成True，color应该是没有变化吧！
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def parse_label():
    # change label to class index    # index2color[0]=(0,0,0)
    color2index[(0,0,0)] = 0 # add an void class  color2index: {(0,0,0):0} ：意思 颜色：000，类别号码：0
    #index2color = {}
    #index2color[0]=[(0,0,0)] #你看后面能不能自己可视化一下
    for obj in labels:  #每个obj都是一类的所有信息，可以“当做”类来使用
        if obj.ignoreInEval:  #“实例.成员函数” ：如果True（如果忽略此标签），那么就忽略了！
            continue
        idx   = obj.trainId  #拿到我们需要的 训练ID
        label = obj.name     #拿到我们需要的 类别名字
        color = obj.color    #拿到我们需要类别的颜色
        color2index[color] = idx  #创建 color作为key，idx作为vaule,#然后color2index就是 dict：key：颜色 value：号码   0~19类
        #index2color[idx] = color  
       


    # parse train, val, test data            #gtfine/train gefine/val gefine/test   #indx的dir
    #test集的color是那种图！！！我就
    for label_dir, index_dir, csv_file in zip([train_dir, val_dir], [train_idx_dir, val_idx_dir], [train_file, val_file]): #删去了test
        f = open(csv_file, "w")
        f.write("img,label\n")
        for city in os.listdir(label_dir):
            city_dir = os.path.join(label_dir, city)
            city_idx_dir = os.path.join(index_dir, city)
            data_dir = city_dir.replace("gtFine", "leftImg8bit") #把gtFine字眼，替换成leftImag8bit字眼 #意味着后来放灰度图？
            if not os.path.exists(city_idx_dir):
                os.makedirs(city_idx_dir)
            for filename in os.listdir(city_dir):
                if 'color' not in filename:  ##filename是city_dir（gtfine）下所有图片
                    continue
                lab_name = os.path.join(city_idx_dir, filename)#//DATA/CityScapes/Labeled_idx/train/ulm/ulm_000076_000019_gtFine_color.png
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png" #ulm_000076_000019_leftImg8bit.png
                img_name = os.path.join(data_dir, img_name)#/home/water/DATA/CityScapes/leftImg8bit/train/ulm/ulm_000076_000019_leftImg8bit.png
                f.write("{},{}.npy\n".format(img_name, lab_name))#写给csv，原图路径，标签图路径

                if os.path.exists(lab_name + '.npy'):
                    print("Skip %s" % (filename))
                    continue
                print("Parse %s" % (filename))

                
                img = os.path.join(city_dir, filename)
                img = imageio.imread(img)#gtfine里，所有带color的图，test的不是！！！test名字对
                img = img[:,:,:3]  #！！！！ debug了一下午，发现这是个4通道图片！！！切！！！！
                height, weight,_ = img.shape
        
                #
                idx_mat = np.zeros((height, weight))
                for h in range(height):
                    for w in range(weight):
                        color = tuple(img[h, w]) #img竟然是个4通道图。。竟然返回 128,0,0,255
                                                # 所以下面全部报错，idx_mat全部是19！！！还好我改了！！！！
                        try:
                            index = color2index[color]
                            idx_mat[h, w] = index
                            #print("finish normal tansform")
                        except:
                            # no index, assign to void
                            #print("no index,make it to 19")
                            idx_mat[h, w] = 19 #没有号的都设置维这个号    
                idx_mat = idx_mat.astype(np.uint8) #转成这个格式
                np.save(lab_name,idx_mat)
                print("Finish %s" % (filename))   #然后png.npy图全部诞生


'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    parse_label()