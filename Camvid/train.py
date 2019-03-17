# -*- coding: utf-8 -*-
from __future__ import print_function  ##这是怕有些人用的python2 兼容python2用的

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler  #Scheduling the learning rate规划学习率
from torch.autograd import Variable  ####注意这里的使用，这个目前已经被淘汰了，官方也有说，改成下面这样就OK
from torch.utils.data import DataLoader  #autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)
from torch.utils.checkpoint import checkpoint

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs   #fcn是自己写的 fcn.py文件
#from Cityscapes_loader import CityscapesDataset   #同
from CamVid_loader import CamVidDataset           #CamVidDataset是自己定义的class数据集 

from matplotlib import pyplot as plt
import numpy as np
import time  #时间模块
import sys   #sys模块包含了与Python解释器和它的环境有关的函数。
import os

####tensorboardx
from logger import Logger
from tensorboardX import SummaryWriter
logger = Logger('/home/water/桌面/wancheng.3.10/logdir')  #使用我们的Logger！！！

#####使用args功能 
import argparse
parser = argparse.ArgumentParser(description='FCN training')
parser.add_argument('--resume', help='checkpoint path')

args = parser.parse_args()


#######################参数设定～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
n_class    = 32  # 数据集类别数

batch_size = 8     #默认4
epochs     = 500   #默认 至少500
lr         = 1e-3  #默认 1e-4
momentum   = 0    #默认 0
w_decay    = 1e-5  #默认1e-5
step_size  = 50   # 50  #用于RMSprop
gamma      = 0.1  # 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)
##############################################################


###和args有冲突吗？？？？冲突就不要argv1，做成Cam专属！！！

#sys.argv[]说白了就是一个从程序外部获取参数的桥梁～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
#我在命令行输入的东西：   python train.py sys.argv[1] sys.argv[2]....sys.argv[n]
## sys.argv[0] :代表 train.py   github作者介绍如何运行这个文件的时候： python train.py CamVid
#if sys.argv[1] == 'CamVid':                     #所以这里是在做选择数据集的工作
root_dir   = "/home/water/DATA/camvid-master/"
#else:
    #root_dir   = "CityScapes/"
train_file = os.path.join(root_dir, "train.csv") #702*0.9张 训练生图和训练序号图.npy ！的路径！是个字符串嘛
val_file   = os.path.join(root_dir, "val.csv")   #702*0.1张 val生图和val序号图.npy  ！的路径！


#给我们的模型建立个路径～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# create dir for model  
model_dir = "models"         #先给它一个名字
if not os.path.exists(model_dir):   #看有没有
    os.makedirs(model_dir)          #makedirs创建一个可以是长串，文件夹下有文件夹的路径，makedir只能在当前路径创建一个文件夹
model_path = os.path.join(model_dir, configs)    #连接dir和congigs？why？？？？？？？？
#  os.path.abspath("__file__") 获取文件绝对路径  (这里应该是和py文件的文件夹里，创一个model文件夹)
                                                 #后面这里还存了 训练好的model


##############设置GPU
#use_gpu = torch.cuda.is_available()                      
# num_gpu = list(range(torch.cuda.device_count())) 我只有1个显卡
#############


############设置一下数据集分割～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
#if sys.argv[1] == 'CamVid':  #训练集
train_data = CamVidDataset(csv_file=train_file, phase='train')
#else:
#    train_data = CityscapesDataset(csv_file=train_file, phase='train')    #网上建议单GPU num_workers = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader_val =DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4) #用于检查是否过拟合
#if sys.argv[1] == 'CamVid':  #val集
val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
#else:
#    val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=4)  #到时候参数自己对着调一调
#########################################


###############设置一下我们的网络模型～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～`
vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class) #n_cliass：分类数
##这里的FCN用的是 FCNs，  8 16 32 自己调试着用                   
#############################


#################把我们的！！模型！！东西送入GPU啦～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～`
 #  device = torch.device('cuda：0' if torch.cuda.is_available() else "cpu") 单引号才可以 cuda:0，不然报错
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #用.to(device)来决定模型使用GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 模型（送成功了） = 模型.to(device)                                    
st = time.time()        #开始计时啦！
vgg_model = vgg_model.to(device)
fcn_model = fcn_model.to(device)
#fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
time_elapsed = time.time() - st
print('Finish cuda loading in {:.4f}m {:.4f}s'.format(time_elapsed // 60, time_elapsed  %60)) # 打印出来时间
                                  #保留小数点后4位            #这个操作用该是分割，匹配前面的 m和s


#!!!!!!!!!!!!!!!!!!!!!!
#～～～～～～～紧随其后，设置下面3样属性，官方教程也是这样的模板   
criterion = nn.BCEWithLogitsLoss()  #  sigmoid+bceloss ，不用output出来再单独加sigmoid了再加bce了
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 50 epochs
        #optimizer是包装好的优化器，step_size是学习率衰减期，指几个epoch衰减一次，gamma是衰减率乘积因子，默认是0.1
###################################################################################



# create dir for score～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
score_dir = os.path.join("scores", configs) #score_dir这里打出来只是个字符串哈
if not os.path.exists(score_dir):
    os.makedirs(score_dir)   #诞生了2个文件夹  socres/configs那一长串
IU_scores    = np.zeros((epochs, n_class))  #epochs=500 ,n_class =20  500*20的零矩阵，10000个0
pixel_scores = np.zeros(epochs)            #一行500个零   #很可能这里是用来存放500轮的数据
###############################


##恢复训练：  测试成功！
if args.resume:
    if os.path.isfile(args.resume):  #?这里报错，换上filename试试
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
            #best_prec1 = checkpoint['best_prec1']
        fcn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        args.start_epoch = checkpoint['epoch']+1
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

##########～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～`
def train():
    fcn_model.train()
    for epoch in range(epochs):   #训练500轮
        scheduler.step() #scheduler.step()：学习率调整！ 一般放在epoch里，看上面是50个epochs执行一次

        st = time.time()  #st = start time
        for iter, batch in enumerate(train_loader):  #train_loader：上面Dataloader加载好的数据
            optimizer.zero_grad()
             #梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积
             #因此这里就需要每个batch设置一遍zero_grad 了。

             #157枚 一枚4张图，batch就是枚举类里的内容
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = batch['X'].to(device)   #X是生图
            labels = batch['Y'].to(device)   #Y是32通道的 one-hot编码后的图

            #作者写的
            #if use_gpu:
                #inputs = Variable(batch['X'].cuda())
                #labels = Variable(batch['Y'].cuda())
            #else:
                #inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)  #把生图送进了我们的FCN，得出一个输出
            #  
            loss = criterion(outputs, labels)  #我们的loss，竟然是和 32通道的one-hot图在作比较
            loss.backward()
            optimizer.step()  #optimizer.step()通常用在每个mini-batch之中，执行优化器参数更新，在我们这4图为一个小batch里运动一下

            if iter % 10 == 0:  #一共训练图总数/batch_size个iter哈 
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item())) #！这里作者给的data[0]，要报错
                
                #试试tensorboardX ，10个iter打一次，也可以放前，每个iter打一次
                info = { 'loss': loss.item() }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter)
                    #tag：key名字，value：值，纵坐标， iter是步长，我这里 500多张训练图，batch=8.一共80个iter左右
                    #所以 一个epoch打 80%10 8次左右
        
        
        #完成了1个epoch
        time_elapsed = time.time() - st
        print('Finish epoch {} in {:.4f}m {:.4f}s'.format(epoch,time_elapsed // 60, time_elapsed  %60))
        #print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        #torch.save(fcn_model, model_path)
        
        torch.save(fcn_model.state_dict(),'/home/water/桌面/wancheng.3.10/statedict/sate.pth')

       # if epoch%5 == 0 : # 每5 epochs 存一下
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': fcn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,     #怎么成功怎么来
            }, 'checkpoint.tar') # ？？#filename='/output/checkpoint.pth.tar' )


        #每个epoch ，save一次，然后还走一次val
        print("This epch is finished,begin val this epoch")
        val(epoch)#  #你看这个 epoch 设置的不错，这样可以实现 每个epoch train一次，val一次
##############################################################################

def val(epoch):  #val就可以当做在测试了！
                      #无论是验证，还是测试，
    fcn_model.eval()  ###必备！！！！！eval（）时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
                                    #不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大。
    total_ious = []       #存放总ious
    pixel_accs = []       #像素精度

    for iter, batch in enumerate(val_loader):  #从val集里读取

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = batch['X'].to(device)   #X是生图
        output = fcn_model(inputs)  # output这里出来是一个 tensor，而且会显示 device = cuda:0
        output = output.data.cpu().numpy()  #转成numpy，必须先把tensor从GPU拿到CPU
                                            

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        #pred ：请看笔记
        #pred:由output转成的灰度图
        #这里把灰度图再转成对应的RGB应该就可以可视化了！
                   
                #tensor.cpu.numpy  #numpy格式的灰度图
        target = batch['l'].cpu().numpy().reshape(N, h, w)  #batch['l'] 是数据集中，生图所对应的序列图 就一张图的点是是0-31，图片是灰黑色
        for p, t in zip(pred, target): #pre和target是2个numpy数组 zip：并行遍历，pred/target两个类别序号图！ 逐点对应！
            total_ious.append(iou(p, t)) #算iou
            pixel_accs.append(pixel_acc(p, t)) #算像素ACC



    #下面这些是 已经经过 iou 和 pixel_acc得到了值以后哈 最终的结果
    #你看下面都是def iou,def acc
    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len，返回（32类别，val_len张图）
    ious = np.nanmean(total_ious, axis=1) #计算沿指定轴的算术平均值，忽略NaN，相当于总loss/val_len张图，返回32个类别的ious
    pixel_accs = np.array(pixel_accs).mean()                                      #下面这是miou（32个ious/32）
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    
    #试试tensorboardX
    accinfo = { 'pixel_accs': pixel_accs,'miou' :np.nanmean(ious)}
    for tag, value in accinfo.items():
        logger.scalar_summary(tag, value, epoch)
        #2图，一个pixelacc一个 miou，每个epoch打一次，一共500epochs

    ##存储我们的iou，acc等信息
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
    print("Finsh val")

# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1)) #max(),返回union和1中，大的那个数（分母肯定要大于1）
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
        #这个print返回每一类的iou供你观察
    return ious  #ious里面是32个浮点数！


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    print("begin to train our model")
    train()
    