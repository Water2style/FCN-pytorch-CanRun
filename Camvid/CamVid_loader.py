# -*- coding: utf-8 -*-
from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import imageio


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir   = "/home/water/DATA/camvid-master"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

num_class = 32
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 720, 960
train_h   = int(h * 2 / 3)  # 480
train_w   = int(w * 2 / 3)  # 640
val_h     = int(h/32) * 32  # 704
val_w     = w               # 960


class CamVidDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5):
        self.data      = pd.read_csv(csv_file) #读取csv文件，csv文件存储着任意字符串
        self.means     = means                #print(data)真的是个表格一样的东西
        self.n_class   = n_class

        self.flip_rate = flip_rate  #后面看要不要随机对训练集进行augment
        self.crop      = crop       #对语义分割数据集使用crop是必须的
        if phase == 'train': 
            self.crop = True       ##作者没有这一行，但是根据人家说，这里可能报错，如果是train
            self.new_h = train_h    #train图片的h，用来后面做crop
            self.new_w = train_w    
        elif phase == 'val':       
            self.flip_rate = 0.     #对验证集不动，可见flip可能用于数据增强
            #self.crop = False       #验证集允许任意大小的图片输入
            self.new_h = val_h      #这个改成了704，可能是为了模仿“任意输入尺寸”
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):              #自动判断的切片  #ix有时候会说被抛弃了，这里改成iloc也行
        img_name   = self.data.iloc[idx, 0]  # 读取的csv.ix[行位置或行标签, 列位置或列标签]
        img        = imageio.imread(img_name) #0是第一列，1是第二列
        label_name = self.data.iloc[idx, 1]     #这里的结果就是: img_name是 训练生图，label_name是对应的标注图npy
        label      = np.load(label_name)     #label_name是哪里，你去看看，是所有 png.npy，所以 np.load!!!!!!
        #这里无法单句运行，还需补补基础！。。反正最后应该是返回：生图和生图所对应的序列图
        #numpy 图像:(高，宽，3通道)
 
        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h) #生成一个 （0~720-480）即（0~240）之间的整数
            left  = random.randint(0, w - self.new_w) #生成一个（0~320）之间的整数
            #裁剪图像 之后是img[高：top到top+480，宽：left到left+640]  （都不会超过720*960）的大小
            #label 对应呀！！ labeled图和img的格式大小对应
            img   = img[top:top + self.new_h, left:left + self.new_w]   
            label = label[top:top + self.new_h, left:left + self.new_w]
            #现在的图片和label图片已经是crop之后的了

        if random.random() < self.flip_rate:#random.random()用于生成一个0到1的随机符点数
            img   = np.fliplr(img)          # <0.5 那么，np.fliplr实现水平镜像处理，形成和原图左右对称的图
            label = np.fliplr(label)      #跟着对称



  
                  #之前 img是[h,w,3]
                  #a是rgb图像，那么
                  #a[::-1]，a[:,::-1]，a[:,:,::-1]分别是X轴的镜像，Y轴的镜像，BGR转换为RGB；
        # reduce mean     #转为bgr之后，颜色变了，hw3没变
        img = img[:, :, ::-1]       # switch to BGR     #可能当时作者编的时候比较老，现在新功能轻易实现这些
                                                   #只有(h,w,3)才能imshow出来
        img = np.transpose(img, (2, 0, 1)) / 255.   #NHWC -> NCHW # transpose：(0,1,2)默认不动，(1,0,2)前2个通道交换，运行一下交换一下
        img[0] -= self.means[0]                     #(2,0,1)是向右平移，运行一下，向右平移一下
        img[1] -= self.means[1]              #这里手动均值化呀 2333333
        img[2] -= self.means[2]   #这里之后，是个 (3,h,w)的numpy图

        # convert to tensor                               
        img = torch.from_numpy(img.copy()).float()  #numpy转tensor ？？？？可能tensor存图的格式要(3,h,w)，最大目的当然是后面用tensor
        label = torch.from_numpy(label.copy()).long() ## label：生图对应的序列图，转tensor

        #独热编码
        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)  # n_class是32类
                                                #torch.Size([32, h, w]) 
        for c in range(self.n_class):
            target[c][label == c] = 1  #target[c]就是地c张zeros，然后判断 label == c，就是标签序号（0-31）等于c的位置
                                       # 那么之后的target，就成了 one-hot编码的32张 0-1图 组成的结果了，可以看看那个图，网上6类的

        sample = {'X': img, 'Y': target, 'l': label} # label是生图对应的序列图(tensor格式)
                  # img： torch.Size([3, h, w])  target：32张one-hot 0-1图，label是生图对应的序列图
        return sample   ## __getitem()__ 所有的所有，就是为了return这个sample
  

def show_batch(batch):                                                                 #batch_size =4
    img_batch = batch['X']   ##根据右面，这个batch是dataloader里，加载完图像之后的，一个batch里的4个图
    img_batch[:,0,...].add_(means[0])   #img_batch 我自己测的时候就只用了1张图，后面运行的多少，自己设置
                                        #这3行就是给means化后的图片还原成彩图
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)  #vscode检测这batch_size没用，确实是
                                        #给图加格子
    grid = utils.make_grid(img_batch)  #torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)[source]在·
    # print(grid)  #是个torch.Tensor
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))  #tensor又转numpy，之前全体右移，现在全体左移回来
                             #[::-1]？                            #成了(h,w,3)，才可以imshow出来
                                                        #？？？？但是为何我画出来！！！颜色不对？？？？
    plt.title('Batch from dataloader')  #给图加标题


if __name__ == "__main__":
    
    train_data = CamVidDataset(csv_file=train_file, phase='train')  #读取的是训练数据

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]                            #看吧 target 32,480,640 是one-hot后的结果
        print(i, sample['X'].size(), sample['Y'].size())  #torch.Size([3, 480, 640]) torch.Size([32, 480, 640])
                                                          #然后这里连着打了4个
    
    
    
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())  #   X:[b_s,3,480,640] Y：[batch_size,32,480,640]
        #我把enumerate里的内容成为“一枚”吧
        #全部打印出来是： 157枚 [4,3,480,640] [4,32,480,640] 相当于一个batch_size就是一枚 一枚里4副图
        # observe 4th batch    #打出第 i个 batch中的4幅图
        if i == 0:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break