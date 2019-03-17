# -*- coding: utf-8 -*-

from __future__ import print_function

import torch                                                 #问题１：为什么转置卷积的参数那么设置
import torch.nn as nn                                        #问题2:5个图怎么做出来的，ranges表的意思？
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG


class FCN32s(nn.Module):
                                                  
    def __init__(self, pretrained_net, n_class):
        super().__init__()  #      #现在只是在定义，我们下面需要用的哪些函数，具体操作要看forward
        self.n_class = n_class
        self.pretrained_net = pretrained_net             #为何通道数要这样设置呀呀呀呀                            
        self.relu    = nn.ReLU(inplace=True)             #参数设定要按照vgg来           #默认daliation是1哈   #这个outpadding看下官方文档
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) #n_class是输出的通道数，转置卷积完了之后，又整了个1*1卷积，目的是为了改变通道数
                                                                              #1*1卷积最大作用：改变通道数=改变参数的多少
    def forward(self, x):
        #每个FCN网络的第一步，都是把图片送入pretrained_net，：VGG RES等
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)  这里的x5,就是经过vgg之后，的  n张，512通道，高/32，宽/32之后的图

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1) 目的是为了改变通道数，这个通道数又怎么设置呢
                                                           #1*1卷积的效果是什么呢？？？？？？  1*1卷积最大作用：改变通道数=改变参数的多少      
        return score  # size=(N, n_class, x.H/1, x.W/1)     #！！！！？？？名字是用来做分类器啊！！！！！！  ？？？？           


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)  ###这个x4 是在vgg中，比x5少经历一次 1/2pool的原图

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):   #看到这里就明白了，我们的x1到x5   x1就是只经历了1次池化的原图，x5经历了5次，缩维为1/32
        output = self.pretrained_net(x)                             
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)              

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)
################################################################################
####################到此为止，就完成了论文里的  skip结构




class VGGNet(VGG):    #这里debug了作者的源代码！  他的：super.__init__(make_layers(cfg[model]))！！ 我的才可以运行！！！
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super(VGGNet,self).__init__(make_layers(cfg[model]))  #cfg[model]是cfg里第三排tuple
        self.ranges = ranges[model]              ##ranges是后面定义的字典，model这里是key，输出对应value
     
                                                     #因为model=vgg16，所以这里ranges是第三排那串，tuple

        if pretrained:                #module.state_dict()返回模型整个状态的字典
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            #exec:动态执行后面的语句    #所以这里。。应该是加载这个pre训练模型吧。。。。                                    
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier    #删除多余的vgg后面的fc

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
                                 #得到x1~x5   5个不同大小的输出图片
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):  #'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)), len:5 
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):   #idx=0 执行5-0次， idx=3 执行17-10次，idx=5 执行 31-24次
                x = self.features[layer](x)  #features：提取网络中各层，他这个就相当于是提取了5,10,17,24,31这5个位置的输出，这5个位置刚好是，maxpool层，让图片缩小1/2的层
            output["x%d"%(idx+1)] = x        #送出分别经历1~5次 池化的图   #不用使劲纠结了，这个算法不是重点

        return output


                                       #ranges不是官方做的表，我还很好奇这5个图是从哪里来的？多看看别人的代码！
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):   ##根据上表，m位置放上maxpool，不是m的位置，64 128 256 512 就作为Conv2d的输出通道数，
    layers = []   #装载nn.conv2d+reulu+bn+conv+............19层信息，为了放入 nn.Sequential
    in_channels = 3
    for v in cfg:
        if v == 'M':                                                                     
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)  #同过for循环制造了一个 layer表，然后当做参数，送给Sequetnital，就不用你打无数段conv，relu啥的了





if __name__ == "__main__":
    batch_size, n_class, h, w = 10, 20, 160, 160  #n_class都自己定的

    # test output size
    vgg_model = VGGNet(requires_grad=True)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))  #单句测试，这个是requires_grad = False的
    #新版本里 torch.autograd.Variable 已经弃用了，直接input = torch.randn(batch_size, 3, 224, 224)就OK
    output = vgg_model(input)   ##input 放进这里，应该就是可以求导的吧
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])   #7 = 224/32
    print("VGG has been checked")
    #assert用法：
       # assert 执行，如果没问题，就啥也不动，如果由问题，返回错误报告

    
    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)                  #检查是否正确地把图片还原
    assert output.size() == torch.Size([batch_size, n_class, h, w])
    print("FCN32S has been checked")

    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])
    print("FCN16S has been checked")

    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])
    print("FCN8S has been checked")

    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, n_class, h, w])
    print("FCNS has been checked")

    print("Pass size check")
    
    
    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class) # jupyter里面print，就是整个论文框架
    criterion = nn.BCELoss()  #BCELOSS的使用，很可能是和one-hot编码有关
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.9)
    #torch.autograd.Variable 新版本里已经被取代，直接后面torch.randn即可
    input = torch.randn(batch_size, 3, h, w)
    y = torch.randn(batch_size, n_class, h, w, requires_grad=False)
    for iter in range(20):
        optimizer.zero_grad()
        output = fcn_model(input)
        #output = nn.functional.sigmoid(output)  #有抛弃警告！
        output = torch.sigmoid(output)   
        loss = criterion(output, y)
        loss.backward()
        print("iter{}, loss {}".format(iter, loss.data))  #loss.data[0] 之前作者写的这个
        optimizer.step()