# -*- coding: utf-8 -*-

from __future__ import print_function

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
root_dir          = "/home/water/DATA/camvid-master"
data_dir          = os.path.join(root_dir, "701_StillsRaw_full")    # train data
label_dir         = os.path.join(root_dir, "LabeledApproved_full")  # train label
label_colors_file = os.path.join(root_dir, "label_colors.txt")      # color to label
val_label_file    = os.path.join(root_dir, "val.csv")               # validation file
train_label_file  = os.path.join(root_dir, "train.csv")             # train file

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}

def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list   = os.listdir(data_dir)  #返回这个目录里，所有内容，‘图1’‘，图2’......
    data_len    = len(data_list)    #702个图片  #注意这里是训练集
    val_len     = int(data_len * val_rate)  #训练集700张，分10%的数量给验证集

    if random_seed:    #设置随机种子
        random.seed(random_seed) #看看后面哪里用

    if shuffle:
        #sample(seq, n) 从序列seq中选择n个随机且独立的元素
        data_idx = random.sample(range(data_len), data_len)  
        # data_idx 是从0到702 随机排序的数组
    else:
        data_idx = list(range(data_len))  #这个就是从0到702 依次排序

    val_idx     = [data_list[i] for i in data_idx[:val_len]] # 前70个，图片名 List
    train_idx   = [data_list[i] for i in data_idx[val_len:]]  # 71到702个


    # !创建 create val.csv
    #  "w"打开一个文件只用于写入。如果该文件已存在则打开文件，
    # 并从开头开始编辑，即原有内容会被删除。
    # 如果该文件不存在，创建新文件。
    v = open(val_label_file, "w")
    v.write("img,label\n") #write() 方法用于向文件中写入指定字符串
    for idx, name in enumerate(val_idx):
        if 'png' not in name:  ##跳过损坏文件
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name.split(".")[0] + "_L.png.npy"
        v.write("{},{}\n".format(img_name, lab_name))
    #最后生成了一个.csv文件，位于根目录
    ## 装的信息是：   2列，一列是验证集，70张  生图路径+名字，第二列是验证集对应的：标签图+名字+.npy
                                      #png.npy ：后面parse_label函数，就是在标签图路径里 生成 标签图+名字+.npy 文件！！！
    # create train.csv                所以这2个.csv文件，这里存放的是信息 ，是： 生图信息和标签图+npy信息
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for idx, name in enumerate(train_idx):
        if 'png' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name)
        lab_name = lab_name.split(".")[0] + "_L.png.npy"
        t.write("{},{}\n".format(img_name, lab_name))



    #parse：分析    分析标签
def parse_label():
    # change label to class index
    #“r”：以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
    #label_colors.txt :！！装的是颜色和对应标签   64 128 64\tAnimal   颜色\t类别
    # 只读，读好了之后                                           #不igore 就会bug
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        label = line.split()[-1]   #提取所有label形成一个字符串 #动物，人，墙..
        color = tuple([int(x) for x in line.split()[:-1]]) #形成一个元组 对应动物，人，墙..
        #的颜色，比如动物的颜色是红色 ：[128,0,0]....
        print(label, color) 
        
        #d[key] = value
        #设置d[key]的值为value，如果该key不存在，则为新增
        #label2color[label] = color 运行后：
        #就形成了1个字典： 以label做key，以color做value的新字典
        #包含内容：{'Animal': (64, 128, 64), 'Archway': (192, 0, 128).....}
        #后面有精彩用法....
        label2color[label] = color  
        color2label[color] = label  #{颜色：标签}
        label2index[label] = idx    # {标签：idx} {'Animal': 0, 'Archway': 1...}
        index2label[idx]   = label  # {idx:标签}
        #下面是作者自己标注的：
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)
                     
                     #enumerate ：迭代器，0号，内容0;1号，内容1
    for idx, name in enumerate(os.listdir(label_dir)):  #os.listdir(label_dir) 是标签集里所有图片
         #idx就是从0开始的序号  name是图片名           #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，这个列表以字母顺序。
        
        filename = os.path.join(label_idx_dir, name)  #  labeled_idx/所有图片名
        if os.path.exists(filename + '.npy'):  #检查是否有图片名.png.npy，当前应该是没有的
            print("Skip %s" % (name))          #有了就跳过这个图 npy是numpy文件
            continue
        print("Parse %s" % (name))   ## 打出：Parse 图片名（不包含路径）
        img = os.path.join(label_dir, name)   ## img是路径，LabeledApproved_full/所有图片名
                                              ## 区分一下 和 filename之间的用法和关联？
        img = imageio.imread(img) #用numpy(npy)格式打开一个图
        height, weight, _ = img.shape  # numpy存储图片格式（高，宽，3通道）
                                       #Tensor是（3，高，宽）
        
 
        #在大for循环里，对每一张图执行下面操作  img是上面读取的一个npy格式的图哈
        idx_mat = np.zeros((height, weight))      #720*960        
        for h in range(height):
            for w in range(weight):        #前面也有个color啊，不同作用域功能不同
                color = tuple(img[h, w])   # tuple(序列)，把序列转为元组
 
                                          #这里应该是把img[h,w]这个！像素点！（128,64,64）
                                           # 抓出来弄成了一个元组，又因为遍历
                                           #所以color是一个有 height*weight个元素的tuple
                           #color包含着这个图片里，所有的颜色          
                try:    #try，except： 异常检测，try里顺序执行，如果，去执行except
                    #tuple类型的color在这里作为key，输出相应的value，也就是label值，dict的存储是一一对应的
                    #所以 出来的label是和输入的color 一一对应
                    label = color2label[color]  # 给彩图像素点，返回像素点的label，就像是上面那图里只有猫和北京，返回：cat space
                    index = label2index[label]  # 给label返回类型代表的号码，给cat sapce，返回1,5
                    idx_mat[h, w] = index  #构成了一个由颜色到标签到标签序号处理后的图，一个点一个点送？
                except:
                    print("error: img:%s, h:%d, w:%d" % (name, h, w))
        idx_mat = idx_mat.astype(np.uint8)   #转换数据类型
        np.save(filename, idx_mat)  #numpy.save(file, arr, allow_pickle=True, fix_imports=True)
                                    #把当前（因为这个for里是逐像素点处理一张图）这个图的信息(numpy)存起来
        print("Finish %s" % (name))
        
    #跳出for，这个位置就是处理好了所有的图，生成了702个 png.npy图
     #生成的这个是一个numpy图，每个图上，是标记好的序号
     #就像 一个张图里是 建筑和空白，建筑位置上显示：4，4 = buildings标签 = buildings颜色[128,0,0]



    # test some pixels' label    ～～～～～～～～～～～～～～～～～～～～～～～～～～`
    #img = os.path.join(label_dir, os.listdir(label_dir)[0])   #img数据：img[height,weight,rgb]
    #img = imageio.imread(img)   
    #test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)] #  img[555,405]:此图此点的！位置信息！
    #test_ans   = ['Car', 'Building', 'Truck_Bus', 'Car']  #这个是肉眼去看哈，看上面的位置，对应的是啥label
    #for idx, t in enumerate(test_cases):
        #color = img[t]     #相当于访问 img上的4个点的位置信息，输出的是这4个点对应的像素值(img是labeled，就那32个规整的颜色)               
        #assert color2label[tuple(color)] == test_ans[idx]  ##检查一下对不对
        #上面是作者乱标的，所以报错，我在jupyter通过肉眼看图并且调试，就对了哈！！



'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)   #mpimg： matplotlib.image 输入的img是个地址哈，不是啥处理后的numpy数组
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
   print("it starts working")
   divide_train_val(random_seed=1)
   parse_label()
   print("process finished")